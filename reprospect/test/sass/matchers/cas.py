import logging
import typing

from reprospect.test.sass.composite import instruction_is, instructions_contain
from reprospect.test.sass.composite_impl import SequenceMatcher
from reprospect.test.sass.controlflow.block import (
    BasicBlockMatcher,
    BasicBlockWithParentMatcher,
)
from reprospect.test.sass.instruction import (
    AddressMatcher,
    AtomicMatcher,
    InstructionMatch,
    LoadGlobalMatcher,
    RegisterMatch,
    RegisterMatcher,
)
from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.test.sass.matchers.move32 import Move32Matcher
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass.controlflow import Graph
from reprospect.tools.sass.decode import RegisterType


class Operation(typing.Protocol):
    def build(self) -> SequenceMatcher:
        ...

class AtomicCASMatcher:
    """
    Find two blocks to prove there is an atomic CAS code path.

    Typically::

        LDG.E.128 R4, desc[UR6][R2.64]
        ...
        IMAD.MOV.U32 R8, RZ, RZ, R4
        DADD R14, R6, UR10
        IMAD.MOV.U32 R9, RZ, RZ, R5
        ...
        DADD R12, R8, UR8
        ...
        IMAD.MOV.U32 R10, RZ, RZ, R6
        IMAD.MOV.U32 R11, RZ, RZ, R7
        ...
        ATOM.E.CAS.128.STRONG.GPU PT, R12, [R2], R8, R12

    Since registers from the load instruction may be moved before they are used,
    this matcher proceeds as follows:

    1. Find the block with a global load, which gives the address register (``[R2]``).
    2. Find the child block that has an atomic CAS using the same address register
       as the global load instruction.
       It gives the register used for the *new value* (``R12``).
    3. The :py:attr:`operation` must be in the same block as the atomic CAS and output
       in the *new value* register, possible *via* moves.
    """
    def __init__(self, arch: NVIDIAArch, operation: Operation, size: int = 64) -> None:
        self.arch: typing.Final[NVIDIAArch] = arch
        self.operation: typing.Final[Operation] = operation #: Operation on the value.
        self.size: typing.Final[int] = size

    def match(self, cfg: Graph) -> list[InstructionMatch] | None: # pylint: disable=too-many-return-statements
        # First, find the block with the global load from memory.
        matcher_ldg = LoadGlobalMatcher(arch=self.arch, size=self.size, readonly=False)
        if (matched := BasicBlockMatcher(matcher=matcher_ldg).match(cfg=cfg)) is None:
            return None
        block_ldg, [matched_ldg] = matched

        if (matched_ldg_src := AddressMatcher(arch=self.arch, memory=MemorySpace.GLOBAL).match(matched_ldg.operands[1])) is None:
            return None

        logging.info(f'Matched {matched_ldg} with address {matched_ldg_src}.')

        # Then, find a child block with an atomic CAS.
        matcher_cas = instructions_contain(instruction_is(AtomicMatcher(
            arch=self.arch,
            operation='CAS',
            dtype=(None, self.size),
            scope='DEVICE',
            consistency='STRONG',
            memory='',
        )).with_operand(index=2, operand=f'[{matched_ldg_src.reg}]'))

        if (matched := BasicBlockWithParentMatcher(parent=block_ldg, matcher=matcher_cas).match(cfg=cfg)) is None:
            return None
        block, [matched_cas] = matched

        assert (matched_cas_new_reg := RegisterMatcher(special=False).match(matched_cas.operands[-1])) is not None
        logging.info(f'Matched {matched_cas} with new value register {matched_cas_new_reg!r}.')
        if matched_cas_new_reg.rtype != RegisterType.GPR:
            return None

        # Find the matching operation.
        if (matched_op := instructions_contain(self.operation.build()).match(instructions=block.instructions[:matcher_cas.next_index])) is None:
            return None

        logging.info(f'Matched operation instructions {matched_op}.')

        # Ensure that the operation outputs relate to the "new value".
        output_regs: typing.Final[tuple[RegisterMatch, ...]] = tuple(
            reg
            for mop in matched_op
            if (reg := RegisterMatcher(special=False).match(mop.operands[0])) is not None
        )

        logging.info(f'Collected operation output registers {output_regs}.')

        if min(output_regs, key=lambda x: x.index).index != matched_cas_new_reg.index:
            for ireg in range(len(output_regs)):
                target = f'{matched_cas_new_reg.rtype}{matched_cas_new_reg.index + ireg}'
                moved = instructions_contain(Move32Matcher(dst=target)).match(
                    instructions=block.instructions[:matcher_cas.next_index],
                )
                if moved is None or moved[0].additional is None:
                    logging.error(f'There is no match for moving a register to {target!r}.')
                    return None
                reg = RegisterMatcher().match(moved[0].additional['src'][0])
                if reg not in output_regs:
                    logging.error(f'Source register of {moved[0]} is not in {output_regs}.')
                    return None

        return [matched_ldg] + matched_op + [matched_cas]
