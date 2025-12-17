"""
CUDA supports polymorphic classes in device code, but virtual function dispatch incurs overhead:

- Direct overhead:
    - vtable lookups (additional instructions and memory traffic)
    - indirect call (register spills/fills, jump)
- Indirect overhead:
    - prevents inlining and other compiler optimizations

To understand and mitigate this overhead, it is of interest to analyze the implementation
of virtual functions on NVIDIA GPUs and the instruction sequence patterns that their use typically
generates in machine code.

:cite:t:zhang-2021 identified the typical instruction sequence pattern for the dynamic dispatch:

- load vtable pointer
- access the vtable to obtain the function offset
- resolve the function address via an additional kernel-specific level of indirection through constant memory
- indirect call

This example implements a virtual function call on device and compares it to static dispatch via a
qualified call. It analyzes resource usage for both dispatch types and then verifies programmatically
the presence of the dynamic dispatch instruction pattern that was identified by :cite:t:zhang-2021.
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
from reprospect.test.sass.instruction import (
    AddressMatcher,
    ConstantMatcher,
    LoadConstantMatcher,
    LoadGlobalMatcher,
    LoadMatcher,
    OpcodeModsMatcher,
)
from reprospect.tools.binaries import ELF, CuObjDump, Function, NVDisasm
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

class DispatchType(StrEnum):
    STATIC  = 'static_foo'
    DYNAMIC = 'dynamic_foo'

class TestDispatch(CMakeAwareTestCase):
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_cuda_virtual_functions_dispatch'

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
    def test(self) -> None:
        """
        Run the executable.
        """
        subprocess.check_call(self.executable)

class TestBinaryAnalysis(TestDispatch):
    """
    Binary analysis.
    """
    BANK_VTABLE: typing.Final[str] = '0x2'

    SIGNATURE: typing.Final[dict[DispatchType, re.Pattern[str]]] = {
        DispatchType.STATIC:  re.compile(r'static_foo_kernel'),
        DispatchType.DYNAMIC: re.compile(r'dynamic_foo_kernel'),
    }

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'examples_cuda_virtual_functions_dispatch.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope='class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file=self.executable, arch=self.arch, sass=True, cwd=self.cwd, cubin=self.cubin.name, demangler=self.demangler)[0]

    @pytest.fixture(scope='class')
    def function(self, cuobjdump: CuObjDump) -> dict[DispatchType, Function]:
        def get_function(dispatch_type: DispatchType) -> Function:
            pattern = self.SIGNATURE[dispatch_type]
            fctn = cuobjdump.functions[next(sig for sig in cuobjdump.functions if re.search(pattern, sig) is not None)]
            logging.info(f'SASS code and resource usage from CuObjDump for {dispatch_type} dispatch type:\n{fctn}.')
            return fctn
        return {dispatch_type: get_function(dispatch_type) for dispatch_type in DispatchType}

    @pytest.fixture(scope='class')
    def decoder(self, function: dict[DispatchType, Function]) -> dict[DispatchType, Decoder]:
        def get_decoder(dispatch_type: DispatchType) -> Decoder:
            decoder = Decoder(code=function[dispatch_type].code)
            logging.info(f'Decoded SASS code for {dispatch_type} dispatch type:\n{dispatch_type}.')
            return decoder
        return {dispatch_type: get_decoder(dispatch_type) for dispatch_type in DispatchType}

class TestResourceUsage(TestBinaryAnalysis):
    """
    Resource usage.
    """
    @pytest.fixture(scope='class')
    def nvdisasm(self, cuobjdump: CuObjDump) -> NVDisasm: # pylint: disable=unused-argument
        return NVDisasm(file=self.cubin, arch=self.arch)

    @pytest.fixture(scope='class')
    def detailed_register_usage(self, function: dict[DispatchType, Function], nvdisasm: NVDisasm) -> dict[DispatchType, typing.Any]:
        def get_registers(dispatch_type: DispatchType) -> typing.Any:
            assert (symbol := function[dispatch_type].symbol) is not None
            nvdisasm.extract_register_usage_from_liveness_range_info(mangled=(symbol,))
            assert (registers := nvdisasm.functions[symbol].registers) is not None
            logging.info(f'Detailed register usage from NVDisasm for {dispatch_type} dispatch type:\n{nvdisasm.functions[symbol]}.')
            return registers
        return {dispatch_type: get_registers(dispatch_type) for dispatch_type in DispatchType}

    def test_resource_usage(self, function: dict[DispatchType, Function]) -> None:
        """
        Verify that dynamic dispatch uses more resources than static dispatch:

        - more general purpose registers
        - stack (for spill/fill)
        - additional constant memory (for vtable)
        """
        assert (resource_usage_static  := function[DispatchType.STATIC ].ru) is not None
        assert (resource_usage_dynamic := function[DispatchType.DYNAMIC].ru) is not None

        # General purpose registers.
        registers_static  = resource_usage_static .register
        registers_dynamic = resource_usage_dynamic.register

        logging.info(f'Register usage for static dispatch type:  {registers_static}.')
        logging.info(f'Register usage for dynamic dispatch type: {registers_dynamic}.')

        assert registers_dynamic > registers_static

        # Stack: dynamic dispatch uses 8 bytes.
        stack_static  = resource_usage_static .stack
        stack_dynamic = resource_usage_dynamic.stack

        logging.info(f'Stack usage for static dispatch type:  {stack_static}.')
        logging.info(f'Stack usage for dynamic dispatch type: {stack_dynamic}.')

        assert stack_static  == 0
        assert stack_dynamic == 8

        # Constant memory: dynamic dispatch has bytes in bank 2 for vtable.
        banks_static  = resource_usage_static .constant
        banks_dynamic = resource_usage_dynamic.constant

        logging.info(f'Constant banks for static dispatch type:  {banks_static}.')
        logging.info(f'Constant banks for dynamic dispatch type: {banks_dynamic}.')

        BANK_VTABLE = int(self.BANK_VTABLE, base=16)
        assert banks_static[0] == banks_dynamic[0]
        assert BANK_VTABLE not in banks_static and BANK_VTABLE in banks_dynamic

    def test_detailed_register_usage(self, detailed_register_usage: dict[DispatchType, typing.Any]) -> None:
        """
        Check detailed register usage (GPR, PRED, UGPR, UPRED) against architecture-dependent expected values.

        Dynamic dispatch uses more general purpose registers (GPR). The use of other register types
        (PRED, UGPR, UPRED) is the same for both dispatch types.
        """
        match self.arch.compute_capability.as_int:
            case 70:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (2, 2)}
            case 75:
                expt_static  = {RegisterType.GPR: (6, 5),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (2, 2), RegisterType.UGPR: (6, 2), RegisterType.UPRED: (1, 1)}
            case 86 | 89:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (2, 2), RegisterType.UGPR: (6, 2), RegisterType.UPRED: (1, 1)}
            case 120:
                expt_static  = {RegisterType.GPR: (8, 7),   RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
                expt_dynamic = {RegisterType.GPR: (24, 18), RegisterType.PRED: (1, 1), RegisterType.UGPR: (6, 2)}
            case _:
                raise ValueError(f'unsupported {self.arch.compute_capability}')

        assert detailed_register_usage[DispatchType.STATIC]  == expt_static
        assert detailed_register_usage[DispatchType.DYNAMIC] == expt_dynamic

class TestSASSDynamic(TestBinaryAnalysis):
    """
    SASS for dynamic dispatch.
    """
    dispatch_type: typing.Final[DispatchType] = DispatchType.DYNAMIC

    @pytest.fixture(scope='class')
    def control_flow(self, decoder: dict[DispatchType, Decoder]) -> Graph:
        """
        Partition SASS into basic blocks.

        The partitioning is expected to result in a basic block that contains the dynamic dispatch instruction
        sequence and a separate basic block with the function implementation, among possibly other blocks.
        """
        cfg = ControlFlow.analyze(instructions=decoder[self.dispatch_type].instructions)
        logging.info(f'Partitioned SASS into {len(cfg.blocks)} basic blocks.')

        # Write out control flow graph as Mermaid diagram.
        ARTIFACT_DIR = pathlib.Path(os.environ['ARTIFACT_DIR'])
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mmd_path = ARTIFACT_DIR / 'example_dispatch.mmd'
        mmd_path.write_text(cfg.to_mermaid())

        print(str(mmd_path))

        return cfg

    @pytest.fixture(scope='class')
    def basic_block_dynamic_call(self, control_flow: Graph) -> BasicBlock:
        """
        Find the basic block that contains the dynamic dispatch instruction sequence by looking
        for a basic block that contains an indirect call instruction.
        """
        matcher_call_rel_noinc = OpcodeModsMatcher(opcode='CALL', modifiers=('REL', 'NOINC'))
        for block in control_flow.blocks:
            if instructions_contain(matcher=matcher_call_rel_noinc).match(block.instructions) is not None:
                basic_block_offset = block.instructions[0].offset
                basic_block_size = len(block.instructions)
                logging.info(f'Found basic block with dynamic call at offset {basic_block_offset} with size {basic_block_size}.')
                return block
        raise AssertionError('Could not find basic block with dynamic call.')

    @pytest.fixture(scope='class')
    def basic_blocks_function_implementations(self, control_flow: Graph) -> list[BasicBlock]:
        """
        Find the basic blocks that contain the function implementations by looking
        for basic blocks that contain a FADD instruction.
        """
        matcher_fadd = OpcodeModsMatcher(opcode='FADD')
        blocks = []
        for block in control_flow.blocks:
            if instructions_contain(matcher=matcher_fadd).match(block.instructions) is not None:
                basic_block_offset = block.instructions[0].offset
                basic_block_size = len(block.instructions)
                logging.info(f'Found basic block with function implementation at offset {basic_block_offset} with size {basic_block_size}.')
                blocks.append(block)
        if not blocks:
            raise AssertionError('Could not find basic block with function implementation.')
        return blocks

    @pytest.fixture(scope='class')
    def constant_bank_vtable(self, function: dict[DispatchType, Function]) -> bytes:
        """
        Read the constant memory bank expected to hold the function address that the dynamic
        dispatch resolves to.
        """
        assert (symbol := function[self.dispatch_type].symbol) is not None
        with ELF(file=self.cubin) as elf:
            section_name = f'.nv.constant{int(self.BANK_VTABLE, base = 16)}.{symbol}'
            section = elf.elf.get_section_by_name(section_name)
            assert section is not None
            return section.data()

    def test_instruction_sequence_pattern_dynamic_call(self, basic_block_dynamic_call: BasicBlock) -> None:
        """
        Verify the presence of the instruction sequence pattern described in :cite:t:zhang-2021
        within the dynamic call basic block.

        The instruction sequence pattern looks like::

            LDC.64 R2, c[0x0][0x380] ;      # Load object pointer (this)
            LDG.E.64 R2, desc[UR4][R2.64] ; # Load vtable pointer (dereference this)
            LD.E R8, desc[UR4][R2.64] ;     # Load function offset from vtable entry
            ...
            LDC.64 R8, c[0x2][R8] ;         # Resolve kernel-specific function address via constant bank
            ...
            CALL.REL.NOINC R10 0x0          # Indirect call
        """
        instructions = basic_block_dynamic_call.instructions

        # Load vtable pointer
        matcher_load_vtable_pointer = instructions_contain(
            matcher=instruction_is(
                LoadGlobalMatcher(arch=self.arch, size=64, readonly=False),
            ),
        )
        assert (matched_load_vtable_pointer := matcher_load_vtable_pointer.match(instructions)) is not None
        vtable_pointer_reg = matched_load_vtable_pointer[0].operands[0]
        logging.info(f'Matched load vtable pointer: {matched_load_vtable_pointer}, destination register: {vtable_pointer_reg}')
        offset = matcher_load_vtable_pointer.next_index

        # Load function offset from vtable entry
        matcher_load_function_offset = instructions_contain(
            matcher=instruction_is(
                LoadMatcher(arch=self.arch, readonly=False, memory=None),
            ).with_operand(
                index=-1,
                operand=AddressMatcher(arch=self.arch, reg=vtable_pointer_reg),
            ),
        )
        assert (matched_load_function_offset := matcher_load_function_offset.match(instructions[offset:])) is not None
        function_offset_reg = matched_load_function_offset[0].operands[0]
        logging.info(f'Matched load function offset: {matched_load_function_offset}, destination register: {function_offset_reg}')
        offset += matcher_load_function_offset.next_index

        # Resolve kernel-specific function address via constant bank
        matcher_load_function_address = instructions_contain(
            matcher=instruction_is(
                LoadConstantMatcher(size=64),
            ).with_operand(
                index=-1,
                operand=ConstantMatcher(bank=self.BANK_VTABLE, offset=function_offset_reg),
            ),
        )
        assert (matched_load_function_address := matcher_load_function_address.match(instructions[offset:])) is not None
        function_address_reg = matched_load_function_address[0].operands[0]
        logging.info(f'Matched load function address: {matched_load_function_address}, destination register: {function_address_reg}')
        offset += matcher_load_function_address.next_index

        # Indirect call
        matcher_indirect_call = instructions_contain(
            matcher=instruction_is(
                OpcodeModsMatcher(opcode='CALL', modifiers=('REL', 'NOINC')),
            ).with_operand(
                index=0,
                operand=f'{function_address_reg} 0x0',
            ),
        )
        assert (matched_indirect_call := matcher_indirect_call.match(instructions[offset:])) is not None
        logging.info(f'Matched indirect call: {matched_indirect_call}')

    def test_constant_bank_vtable(self, basic_blocks_function_implementations: list[BasicBlock], constant_bank_vtable: bytes) -> None:
        """
        Show that the kernel-specific function address resolution via the constant bank as in::

            LDC.64 R8, c[0x2][R8] ;

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

    def test_basic_blocks_function_implementations(self, basic_blocks_function_implementations: list[BasicBlock]) -> None:
        """
        The SASS code for the ``dynamic_foo_kernel`` contains not only basic blocks with the implementations
        of ``DerivedA::foo(unsigned int)`` and ``DerivedB::foo(unsigned int)``, but also basic blocks with the
        implementations of ``DerivedA::bar(unsigned int)`` and ``DerivedB::bar(unsigned int)``, even though the
        latter are not called. In fact, in the SASS code, each kernel appears to contain the implementations of all
        virtual functions of all derived classes in the compile unit, even those that are sure not to be called.
        """
        logging.info(f'Found {len(basic_blocks_function_implementations)} basic blocks with function implementations.')
        assert len(basic_blocks_function_implementations) == 4

        matcher_fadd_derived_a_bar = instructions_contain(
            matcher=instruction_is(
                OpcodeModsMatcher(opcode='FADD'),
            ).with_operand(index=-1, operand=str(int('0xab', base=16))),
        )
        assert any(matcher_fadd_derived_a_bar.match(block.instructions) is not None for block in basic_blocks_function_implementations)
