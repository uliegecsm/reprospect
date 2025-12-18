import logging
import typing

from reprospect.test.sass.composite import instruction_is, interleaved_instructions_are
from reprospect.test.sass.composite_impl import SequenceMatcher
from reprospect.test.sass.controlflow.block import BasicBlockMatcher, BasicBlockWithParentMatcher
from reprospect.test.sass.instruction import (
    AtomicMatcher,
    InstructionMatch,
    LoadGlobalMatcher,
)
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass.controlflow import Graph

class Operation(typing.Protocol):
    def build(self, loads: typing.Collection[InstructionMatch]) -> SequenceMatcher:
        ...

class AtomicCAS:
    """
    Find two blocks to prove there is an atomic CAS code path.

    The first block loads the value from global memory.

    The second block performs the :py:attr:`operation` followed by an atomic CAS.

    Typically::

        LDG.E.64 R4, desc[UR6][R8.64]
        ...
        FADD R6, R4, UR10
        FADD R7, R5, UR11
        ...
        ATOM.E.CAS.64.STRONG.GPU P0, R2, [R8], R4, R6
    """
    def __init__(self, arch: NVIDIAArch, operation: Operation, size: int = 64) -> None:
        self.arch: typing.Final[NVIDIAArch] = arch
        self.operation: typing.Final[Operation] = operation #: Operation on the value.
        self.size: typing.Final[int] = size

    def match(self, cfg: Graph) -> list[InstructionMatch] | None:
        matches: list[InstructionMatch] = []

        # First, find the global load from memory.
        matcher_ldg = LoadGlobalMatcher(arch=self.arch, size=self.size, readonly=False)
        if (matched := BasicBlockMatcher(matcher=matcher_ldg).match(cfg=cfg)) is None:
            return None
        block_ldg, [matched_ldg] = matched
        old_value_reg = matched_ldg.operands[0]
        logging.info(f'Matched old value load {matched_ldg}, with destination register {old_value_reg!r}.')

        matches.append(matched_ldg)

        # Then, match the operation with the CAS.
        # Require that the "compare" register (containing the old value for comparison)
        # is the "destination" register of the preceding global load.
        matcher_op = self.operation.build(loads=(matched_ldg,))
        matcher_cas = instruction_is(AtomicMatcher(
            arch=self.arch,
            operation='CAS',
            dtype=(None, self.size),
            scope='DEVICE',
            consistency='STRONG',
            memory='',
        )).with_operand(index=3, operand=old_value_reg)

        if (matched := BasicBlockWithParentMatcher(
            parent=block_ldg,
            matcher=interleaved_instructions_are(matcher_op, matcher_cas),
        ).match(cfg=cfg)) is None:
            return None

        assert len(matched[1]) >= 2

        return matches + matched[1]
