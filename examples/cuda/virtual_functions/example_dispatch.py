"""
CUDA supports polymorphic classes in device code. However, dynamic dispatch for virtual functions
incurs overhead:

- Direct overhead:

    - *vtable* lookups incur additional instructions and memory traffic
    - indirect call may cause register spills/fills and a jump

- Indirect overhead:

    - prevents inlining and other compiler optimizations

:cite:t:`zhang-2021` identified the typical instruction sequence pattern
that dynamic dispatch for virtual functions typically generates in machine code:

- load *vtable* pointer
- access the *vtable* to obtain the function offset
- resolve the function address via an additional kernel-specific level of indirection through constant memory
- indirect call

This example compares dynamic dispatch for a virtual function call on device with static dispatch.
It analyzes resource usage for both dispatch types and then verifies programmatically
the presence of the dynamic dispatch instruction pattern that was identified by :cite:t:`zhang-2021`.

In this way, this example demonstrates how to use `ReProspect` to create research artifacts
that may accompany publications on CUDA code analysis.
"""

import logging
import os
import pathlib
import re
import struct
import subprocess
import sys
import typing

import pytest

from reprospect.test.case import CMakeAwareTestCase
from reprospect.test.sass.composite import instruction_is, instructions_contain
from reprospect.test.sass.controlflow.block import BasicBlockMatcher
from reprospect.test.sass.instruction import (
    AddressMatcher,
    ConstantMatcher,
    Fp32AddMatcher,
    LoadConstantMatcher,
    LoadGlobalMatcher,
    LoadMatcher,
    OpcodeModsMatcher,
)
from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.tools.binaries import (
    ELF,
    CuObjDump,
    DetailedRegisterUsage,
    Function,
    NVDisasm,
    ResourceUsage,
)
from reprospect.tools.sass import ControlFlow, Decoder
from reprospect.tools.sass.controlflow import BasicBlock, Graph
from reprospect.tools.sass.decode import RegisterType
from reprospect.utils import detect

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Dispatch(StrEnum):
    STATIC  = 'static'
    DYNAMIC = 'dynamic'

class MemberFunction(StrEnum):
    FOO = 'foo'
    BAR = 'bar'

class Derived(StrEnum):
    DERIVED_A = 'DerivedA'
    DERIVED_B = 'DerivedB'

class TestDispatch(CMakeAwareTestCase):
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_cuda_virtual_functions_dispatch'

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
    def test_run(self) -> None:
        subprocess.check_call(self.executable)

class TestBinaryAnalysis(TestDispatch):
    SIGNATURE: typing.Final[dict[tuple[Dispatch, MemberFunction], re.Pattern[str]]] = {
        (Dispatch.STATIC, MemberFunction.FOO):  re.compile(rf'{Dispatch.STATIC}_{MemberFunction.FOO}_kernel'),
        (Dispatch.DYNAMIC, MemberFunction.FOO): re.compile(rf'{Dispatch.DYNAMIC}_{MemberFunction.FOO}_kernel'),
        (Dispatch.DYNAMIC, MemberFunction.BAR): re.compile(rf'{Dispatch.DYNAMIC}_{MemberFunction.BAR}_kernel'),
    }

    MARKER: typing.Final[dict[tuple[Derived, MemberFunction], int]] = {
        (Derived.DERIVED_A, MemberFunction.FOO): 0xaf,
        (Derived.DERIVED_B, MemberFunction.FOO): 0xbf,
        (Derived.DERIVED_A, MemberFunction.BAR): 0xab,
        (Derived.DERIVED_B, MemberFunction.BAR): 0xbb,
    }

    BANK_VTABLE: typing.Final[str] = '0x2'

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'{self.get_target_name()}.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope='class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(
            file=self.executable, arch=self.arch,
            sass=True,
            cwd=self.cwd, cubin=self.cubin.name,
            demangler=self.demangler,
        )[0]

    @pytest.fixture(scope='class')
    def function(self, cuobjdump: CuObjDump) -> dict[tuple[Dispatch, MemberFunction], Function]:
        """
        Collect the SASS code and parse the resource usage information.
        """
        def get_function(dispatch: Dispatch, member_function: MemberFunction) -> Function:
            pattern = self.SIGNATURE[dispatch, member_function]
            fctn = cuobjdump.functions[next(sig for sig in cuobjdump.functions if pattern.search(sig) is not None)]
            logging.info(f'SASS code and resource usage from CuObjDump for {dispatch} {member_function} kernel:\n{fctn}')
            return fctn
        return {(dispatch, member_function): get_function(dispatch, member_function)
                for (dispatch, member_function) in self.SIGNATURE}

    @pytest.fixture(scope='class')
    def decoder(self, function: dict[tuple[Dispatch, MemberFunction], Function]) -> dict[tuple[Dispatch, MemberFunction], Decoder]:
        def get_decoder(dispatch: Dispatch, member_function: MemberFunction) -> Decoder:
            decoder = Decoder(code=function[dispatch, member_function].code)
            logging.info(f'Decoded SASS code for {dispatch} {member_function} kernel:\n{decoder}.')
            return decoder
        return {(dispatch, member_function): get_decoder(dispatch, member_function)
                for (dispatch, member_function) in self.SIGNATURE}

class TestResourceUsage(TestBinaryAnalysis):
    """
    Compare resource usage between :py:attr:`Dispatch.STATIC` and :py:attr:`Dispatch.DYNAMIC`
    dispatch for :py:attr:`MemberFunction.FOO`.
    """
    MEMBER_FUNCTION: typing.Final[MemberFunction] = MemberFunction.FOO

    @pytest.fixture(scope='class')
    def static(self, function: dict[tuple[Dispatch, MemberFunction], Function]) -> ResourceUsage:
        assert (ru := function[Dispatch.STATIC, self.MEMBER_FUNCTION].ru) is not None
        return ru

    @pytest.fixture(scope='class')
    def dynamic(self, function: dict[tuple[Dispatch, MemberFunction], Function]) -> ResourceUsage:
        assert (ru := function[Dispatch.DYNAMIC, self.MEMBER_FUNCTION].ru) is not None
        return ru

    @pytest.fixture(scope='class')
    def nvdisasm(self, cuobjdump: CuObjDump) -> NVDisasm:
        return NVDisasm(file=cuobjdump.file, arch=self.arch)

    @pytest.fixture(scope='class')
    def detailed_register_usage(self, function: dict[tuple[Dispatch, MemberFunction], Function], nvdisasm: NVDisasm) -> dict[Dispatch, DetailedRegisterUsage]:
        def get_registers(dispatch: Dispatch) -> DetailedRegisterUsage:
            assert (symbol := function[dispatch, self.MEMBER_FUNCTION].symbol) is not None
            nvdisasm.extract_register_usage_from_liveness_range_info(mangled=(symbol,))
            assert (registers := nvdisasm.functions[symbol].registers) is not None
            logging.info(f'Detailed register usage from NVDisasm for {dispatch} {self.MEMBER_FUNCTION} kernel:\n{nvdisasm.functions[symbol]}.')
            return registers
        return {dispatch: get_registers(dispatch) for dispatch in Dispatch}

    def test_dynamic_uses_more_gprs(self, static: ResourceUsage, dynamic: ResourceUsage) -> None:
        logging.info(f'General purpose register usage for {Dispatch.STATIC}  dispatch: {static.register}.')
        logging.info(f'General purpose register usage for {Dispatch.DYNAMIC} dispatch: {dynamic.register}.')

        assert dynamic.register > static.register

    def test_dynamic_uses_stack(self, static: ResourceUsage, dynamic: ResourceUsage) -> None:
        logging.info(f'Stack usage for {Dispatch.STATIC}  dispatch: {static.stack}.')
        logging.info(f'Stack usage for {Dispatch.DYNAMIC} dispatch: {dynamic.stack}.')

        assert static.stack  == 0
        assert dynamic.stack == 8

    def test_dynamic_has_bytes_in_constant_bank_for_vtable(self, static: ResourceUsage, dynamic: ResourceUsage) -> None:
        logging.info(f'Constant bank usage for {Dispatch.STATIC}  dispatch: {static.constant}.')
        logging.info(f'Constant bank usage for {Dispatch.DYNAMIC} dispatch: {dynamic.constant}.')

        BANK_VTABLE = int(self.BANK_VTABLE, base=16)
        assert static.constant[0] == dynamic.constant[0]
        assert BANK_VTABLE not in static.constant and BANK_VTABLE in dynamic.constant

    def test_dynamic_uses_more_registers(self, detailed_register_usage: dict[Dispatch, DetailedRegisterUsage]) -> None:
        match self.arch.compute_capability.as_int:
            case 70:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (2, 2)}
            case 75:
                expt_static  = {RegisterType.GPR: (6, 5),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
            case 86:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (2, 2), RegisterType.UGPR: (6, 2), RegisterType.UPRED: (1, 1)}
            case 89 | 90:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
            case 100:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (1, 1), RegisterType.UGPR: (8, 4), RegisterType.UPRED: (1, 1)}
            case 120:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
            case _:
                raise ValueError(f'unsupported {self.arch.compute_capability}')

        assert detailed_register_usage[Dispatch.STATIC]  == expt_static
        assert detailed_register_usage[Dispatch.DYNAMIC] == expt_dynamic

class TestDynamicDispatchInstructionSequence(TestBinaryAnalysis):
    """
    Look for the dynamic dispatch instruction pattern identified by :cite:t:`zhang-2021`.
    """
    DISPATCH: typing.Final[Dispatch] = Dispatch.DYNAMIC
    MEMBER_FUNCTION: typing.Final[MemberFunction] = MemberFunction.FOO

    @pytest.fixture(scope='class')
    def cfg(self, decoder: dict[tuple[Dispatch, MemberFunction], Decoder]) -> Graph:
        cfg = ControlFlow.analyze(instructions=decoder[self.DISPATCH, self.MEMBER_FUNCTION].instructions)
        logging.info(f'Partitioned SASS into {len(cfg.blocks)} basic blocks.')

        # Write out control flow graph as Mermaid diagram.
        ARTIFACT_DIR = pathlib.Path(os.environ['ARTIFACT_DIR'])
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mmd_path = ARTIFACT_DIR / 'example_dispatch.mmd'
        mmd_path.write_text(cfg.to_mermaid())

        return cfg

    @pytest.fixture(scope='class')
    def basic_block_dynamic_call(self, cfg: Graph) -> BasicBlock:
        """
        Find the basic block that contains the dynamic dispatch instruction sequence by looking
        for a basic block that contains an indirect call instruction.
        """
        matcher_call_rel_noinc = OpcodeModsMatcher(opcode='CALL', modifiers=('REL', 'NOINC'))
        block, _ = BasicBlockMatcher(matcher=matcher_call_rel_noinc).assert_matches(cfg)
        logging.info(f'Found basic block with dynamic call at offset {block.instructions[0].offset} with size {len(block.instructions)}.')
        return block

    @pytest.fixture(scope='class')
    def basic_blocks_function_implementations(self, cfg: Graph) -> tuple[BasicBlock, ...]:
        """
        Find the basic blocks that contain the function implementations by looking
        for basic blocks that contain a `FADD` instruction with the expected operand.
        """
        def get_basic_block(derived: Derived) -> BasicBlock:
            marker = self.MARKER[derived, self.MEMBER_FUNCTION]
            matcher = instructions_contain(
                matcher=instruction_is(OpcodeModsMatcher(opcode='FADD')).with_operand(index=-1, operand=str(marker)),
            )
            block, _ = BasicBlockMatcher(matcher=matcher).assert_matches(cfg)
            logging.info(
                f'Found basic block with {derived}::{self.MEMBER_FUNCTION} implementation '
                f'at offset {block.instructions[0].offset} with size {len(block.instructions)}.',
            )
            return block
        return tuple(get_basic_block(derived) for derived in Derived)

    @pytest.fixture(scope='class')
    def constant_bank_vtable(self, cuobjdump: CuObjDump, function: dict[tuple[Dispatch, MemberFunction], Function]) -> bytes:
        """
        Read the constant memory bank expected to hold the function address that the dynamic
        dispatch resolves to.
        """
        assert (symbol := function[self.DISPATCH, self.MEMBER_FUNCTION].symbol) is not None
        with ELF(file=cuobjdump.file) as elf:
            section_name = f'.nv.constant{int(self.BANK_VTABLE, base=16)}.{symbol}'
            section = elf.elf.get_section_by_name(section_name)
            assert section is not None
            return section.data()

    def test_instruction_sequence_pattern_dynamic_call(self, basic_block_dynamic_call: BasicBlock) -> None:
        """
        Verify the presence of the instruction sequence pattern described in :cite:t:`zhang-2021`
        within the dynamic call basic block.

        The instruction sequence pattern looks like::

            LDC.64 R2, c[0x0][0x380]       # Load object pointer (this)
            LDG.E.64 R2, desc[UR4][R2.64]  # Load vtable pointer (dereference this)
            LD.E R8, desc[UR4][R2.64]      # Load function offset from vtable
            ...
            LDC.64 R8, c[0x2][R8]          # Resolve kernel-specific function address via constant bank
            ...
            CALL.REL.NOINC R8 0x0          # Indirect call
        """
        # Load vtable pointer.
        matcher_load_vtable_pointer = instructions_contain(matcher=instruction_is(
                LoadGlobalMatcher(arch=self.arch, size=64, readonly=False,
        )))
        matched_load_vtable_pointer = matcher_load_vtable_pointer.assert_matches(basic_block_dynamic_call.instructions)
        vtable_pointer_reg = matched_load_vtable_pointer[0].operands[0]
        logging.info(f'Matched load vtable pointer: {matched_load_vtable_pointer}, destination register: {vtable_pointer_reg}')
        offset = matcher_load_vtable_pointer.next_index

        # Load function offset from vtable entry.
        matcher_load_function_offset = instructions_contain(
            matcher=instruction_is(
                LoadMatcher(arch=self.arch, readonly=False, memory=MemorySpace.GENERIC),
            ).with_operand(
                index=-1,
                operand=AddressMatcher(arch=self.arch, memory=MemorySpace.GENERIC, reg=vtable_pointer_reg),
            ),
        )
        matched_load_function_offset = matcher_load_function_offset.assert_matches(basic_block_dynamic_call.instructions[offset:])
        function_offset_reg = matched_load_function_offset[0].operands[0]
        logging.info(f'Matched load function offset: {matched_load_function_offset}, destination register: {function_offset_reg}')
        offset += matcher_load_function_offset.next_index

        # Resolve kernel-specific function address via constant bank.
        matcher_load_function_address = instructions_contain(
            matcher=instruction_is(
                LoadConstantMatcher(size=64),
            ).with_operand(
                index=-1,
                operand=ConstantMatcher(bank=self.BANK_VTABLE, offset=function_offset_reg),
            ),
        )
        matched_load_function_address = matcher_load_function_address.assert_matches(basic_block_dynamic_call.instructions[offset:])
        function_address_reg = matched_load_function_address[0].operands[0]
        logging.info(f'Matched load function address: {matched_load_function_address}, destination register: {function_address_reg}')
        offset += matcher_load_function_address.next_index

        # Indirect call.
        matcher_indirect_call = instructions_contain(
            matcher=instruction_is(
                OpcodeModsMatcher(opcode='CALL', modifiers=('REL', 'NOINC')),
            ).with_operand(
                index=0,
                operand=f'{function_address_reg}',
            ),
        )
        matched_indirect_call = matcher_indirect_call.assert_matches(basic_block_dynamic_call.instructions[offset:])
        logging.info(f'Matched indirect call: {matched_indirect_call}')

    def test_constant_bank_vtable(self, basic_blocks_function_implementations: tuple[BasicBlock, ...], constant_bank_vtable: bytes) -> None:
        """
        Show that the kernel-specific function address resolution via the constant bank as in::

            LDC.64 R8, c[0x2][R8]

        may indeed resolve the indirect call as in::

            CALL.REL.NOINC R10 0x0

        to the function implementation basic block. This is done by verifying that each function
        implementation basic block offset is among the entries held in the constant bank.
        """
        entry_count = len(constant_bank_vtable) // 8
        entries = struct.unpack(f'<{entry_count}Q', constant_bank_vtable)
        logging.info(f'Unpacked constant bank {self.BANK_VTABLE} as array of {entry_count} 8-byte entries:\n{entries}')

        for block in basic_blocks_function_implementations:
            block_offset = block.instructions[0].offset
            assert block_offset in entries

class TestVtableLookupFooVsBar(TestBinaryAnalysis):
    """
    Compare the instructions generated for :code:`dynamic_foo_kernel` and :code:`dynamic_bar_kernel`.
    """
    DISPATCH: typing.Final[Dispatch] = Dispatch.DYNAMIC

    def test_all_instructions_identical_except_load_function_offset(self, decoder: dict[tuple[Dispatch, MemberFunction], Decoder]) -> None:
        """
        All instructions are identical between the two kernels, except for the instruction that
        loads the function offset from the *vtable*, which differs by an 8-byte memory address offset.

        Whereas the instruction that loads the function offset from *vtable* looks for :code:`dynamic_foo_kernel` like::

            LD.E R8, desc[UR4][R2.64]

        it looks for :code:`dynamic_bar_kernel` like::

            LD.E R8, desc[UR4][R2.64+0x8]
        """
        instructions_foo = decoder[self.DISPATCH, MemberFunction.FOO].instructions
        instructions_bar = decoder[self.DISPATCH, MemberFunction.BAR].instructions

        assert len(instructions_foo) == len(instructions_bar)

        # There is exactly one instruction that differs.
        [(instr_foo, instr_bar)] = [(instr_foo, instr_bar) for instr_foo, instr_bar in zip(instructions_foo, instructions_bar, strict=True) if instr_foo != instr_bar]

        # The differing instruction is a load from generic memory.
        matcher_load = LoadMatcher(arch=self.arch, readonly=False, memory=MemorySpace.GENERIC)
        assert (matched_load_foo := matcher_load.match(instr_foo)) is not None
        assert (matched_load_bar := matcher_load.match(instr_bar)) is not None

        # The memory address differs by an 8-byte offset.
        matcher_address = AddressMatcher(arch=self.arch, memory=MemorySpace.GENERIC)
        assert (matched_address_load_foo := matcher_address.match(matched_load_foo.operands[1])) is not None
        assert (matched_address_load_bar := matcher_address.match(matched_load_bar.operands[1])) is not None

        assert matched_address_load_foo.reg == matched_address_load_bar.reg

        assert matched_address_load_foo.offset is None
        assert isinstance(matched_address_load_bar.offset, str)
        assert matched_address_load_bar.offset == '0x8'

class TestAllImplementationsAreInAllKernels(TestBinaryAnalysis):
    """
    :py:meth:`TestVtableLookupFooVsBar.test_all_instructions_identical_except_load_function_offset` shows
    the instructions generated for :code:`dynamic_foo_kernel` and :code:`dynamic_bar_kernel` are identical except
    for the instruction that loads the function offset from the *vtable*.
    This finding is surprising because it indicates that the SASS codes for each kernel must each contain
    the implementations of both virtual functions, even though each kernel only calls one of the virtual functions.

    In fact, when inspecting the SASS code, it can be observed that the SASS code for each kernel contains the implementations
    of all virtual member functions of all derived classes in the compile unit.
    """
    DISPATCH: typing.Final[Dispatch] = Dispatch.DYNAMIC

    def test(self, decoder: dict[tuple[Dispatch, MemberFunction], Decoder]) -> None:
        """
        Assert that the implementations of

        - :code:`DerivedA::foo(unsigned int)`
        - :code:`DerivedB::foo(unsigned int)`
        - :code:`DerivedA::bar(unsigned int)`
        - :code:`DerivedB::bar(unsigned int)`

        are present in the SASS code of each kernel.
        """
        matcher_fadd = Fp32AddMatcher()

        for member_function in MemberFunction:
            cfg = ControlFlow.analyze(instructions=decoder[self.DISPATCH, member_function].instructions)

            basic_blocks: list[BasicBlock] = []
            for marker in self.MARKER.values():
                block, _ = BasicBlockMatcher(
                    matcher=instructions_contain(
                        matcher=instruction_is(matcher_fadd).with_operand(index=-1, operand=str(marker)),
                    ),
                ).assert_matches(cfg)
                basic_blocks.append(block)
                logging.info(f'Found function implementation with operand {marker:#x} in basic block at offset {block.instructions[0].offset} in {self.DISPATCH} {member_function} kernel.')

            assert len(set(basic_blocks)) == len(self.MARKER)
