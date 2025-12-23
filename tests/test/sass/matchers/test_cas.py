import typing

import pytest

from reprospect.test.sass.composite import interleaved_instructions_are
from reprospect.test.sass.composite_impl import SequenceMatcher
from reprospect.test.sass.instruction import Fp32AddMatcher
from reprospect.test.sass.matchers.cas import AtomicCASMatcher
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass.controlflow import BasicBlock, Graph
from reprospect.tools.sass.decode import ControlCode, Instruction


class Operation:
    def build(self) -> SequenceMatcher:
        return interleaved_instructions_are(Fp32AddMatcher(), Fp32AddMatcher())

class TestAtomicCASMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.matchers.cas.AtomicCASMatcher`.
    """
    CONTROL_CODE: typing.Final[ControlCode] = ControlCode.decode(code='0x000e220000000800')

    BLOCK_LDG: typing.Final[BasicBlock] = BasicBlock(instructions=(
        Instruction(offset=0, instruction='LDG.E.64 R4, desc[UR6][R8.64]', hex='0x0', control=CONTROL_CODE),
    ))

    BLOCK_CAS: typing.Final[BasicBlock] = BasicBlock(instructions=(
        Instruction(offset=1, instruction='BSSY B1, 0x300', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=2, instruction='FADD R6, R4, UR8', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=3, instruction='FADD R7, R5, UR9', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=4, instruction='YIELD', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=5, instruction='ATOM.E.CAS.64.STRONG.GPU P0, R2, [R8], R4, R6', hex='0x0', control=CONTROL_CODE),
    ))

    BLOCK_CAS_MOV: typing.Final[BasicBlock] = BasicBlock(instructions=(
        Instruction(offset=1, instruction='BMOV.32.CLEAR RZ, B1', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=2, instruction='BSSY B1, 0x390', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=3, instruction='FADD R17, R4, c[0x0][0x188]', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=4, instruction='FADD R12, R5, c[0x0][0x18c]', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=5, instruction='IMAD.MOV.U32 R6, RZ, RZ, R17', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=6, instruction='IMAD.MOV.U32 R7, RZ, RZ, R12', hex='0x0', control=CONTROL_CODE),
        Instruction(offset=7, instruction='ATOM.E.CAS.64.STRONG.GPU P0, R6, [R8], R4, R6', hex='0x0', control=CONTROL_CODE),
    ))

    CFG:     typing.Final[Graph] = Graph(blocks=[BLOCK_LDG, BLOCK_CAS],     edges={BLOCK_LDG: {BLOCK_CAS}})
    CFG_MOV: typing.Final[Graph] = Graph(blocks=[BLOCK_LDG, BLOCK_CAS_MOV], edges={BLOCK_LDG: {BLOCK_CAS_MOV}})

    @pytest.mark.parametrize('cfg', [CFG, CFG_MOV])
    def test(self, cfg: Graph) -> None:
        assert (matched := AtomicCASMatcher(
            arch=NVIDIAArch.from_compute_capability(90),
            operation=Operation(),
            size=64,
        ).match(cfg=cfg)) is not None
        assert len(matched) == 4
