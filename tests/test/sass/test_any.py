import logging
import typing

import pytest

from reprospect.test.sass.instruction import AnyMatcher, InstructionMatch


class TestAnyMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.AnyMatcher`.
    """
    INSTRUCTIONS: typing.Final[dict[str, InstructionMatch]] = {
        'IMAD.MOV.U32 R4, R4, R5, R6': InstructionMatch(opcode='IMAD', modifiers=('MOV', 'U32'), operands=('R4', 'R4', 'R5', 'R6')),
        'FADD.FTZ.RN R0, R1, R2': InstructionMatch(opcode='FADD', modifiers=('FTZ', 'RN'), operands=('R0', 'R1', 'R2')),
        'MOV R8, c[0x0][0x140]': InstructionMatch(opcode='MOV', modifiers=(), operands=('R8', 'c[0x0][0x140]')),
        'LDG.E.SYS R4, [R2]': InstructionMatch(opcode='LDG', modifiers=('E', 'SYS'), operands=('R4', '[R2]')),
        'STG.E [R2], R4': InstructionMatch(opcode='STG', modifiers=('E',), operands=('[R2]', 'R4')),
        'STG.E.64.SYS [R2.64+UR4], R4': InstructionMatch(opcode='STG', modifiers=('E', '64', 'SYS'), operands=('[R2.64+UR4]', 'R4')),
        'IADD3 R2, R2, 0x4, RZ': InstructionMatch(opcode='IADD3', modifiers=(), operands=('R2', 'R2', '0x4', 'RZ')),
        'ISETP.NE.AND P0, PT, R1, RZ, PT': InstructionMatch(opcode='ISETP', modifiers=('NE', 'AND'), operands=('P0', 'PT', 'R1', 'RZ', 'PT')),
        'FMUL.FTZ R2, R2, R3': InstructionMatch(opcode='FMUL', modifiers=('FTZ',), operands=('R2', 'R2', 'R3')),
        'S2R R0, SR_CTAID.X': InstructionMatch(opcode='S2R', modifiers=(), operands=('R0', 'SR_CTAID.X')),
        'BRA 0x240': InstructionMatch(opcode='BRA', modifiers=(), operands=('0x240',)),
        'EXIT': InstructionMatch(opcode='EXIT', modifiers=(), operands=()),
        'NOP': InstructionMatch(opcode='NOP', modifiers=(), operands=()),
        'FMUL R6, R27.reuse, 0.044714998453855514526': InstructionMatch(opcode='FMUL', modifiers=(), operands=('R6', 'R27.reuse', '0.044714998453855514526')),
        'F2FP.BF16.PACK_AB R27, R6, R27': InstructionMatch(opcode='F2FP', modifiers=('BF16', 'PACK_AB'), operands=('R27', 'R6', 'R27')),
        'DADD R12, |R12|, |R14|': InstructionMatch(opcode='DADD', modifiers=(), operands=('R12', '|R12|', '|R14|')),
        'LDG.E R0, desc[UR12][R18.64]': InstructionMatch(opcode='LDG', modifiers=('E',), operands=('R0', 'desc[UR12][R18.64]')),
        'LDS.64 R12, [UR7+0x8]': InstructionMatch(opcode='LDS', modifiers=('64',), operands=('R12', '[UR7+0x8]')),
        'HFMA2 R7, -RZ, RZ, 0, 5.9604644775390625e-08': InstructionMatch(opcode='HFMA2', modifiers=(), operands=('R7', '-RZ', 'RZ', '0', '5.9604644775390625e-08')),
        'ATOMG.E.ADD.STRONG.GPU PT, R4, desc[UR16][R4.64], R7': InstructionMatch(opcode='ATOMG', modifiers=('E', 'ADD', 'STRONG', 'GPU'), operands=('PT', 'R4', 'desc[UR16][R4.64]', 'R7')),
        'MEMBAR.SC.GPU': InstructionMatch(opcode='MEMBAR', modifiers=('SC', 'GPU'), operands=()),
        'LOP3.LUT P0, RZ, R8, R94, RZ, 0xfc, !PT': InstructionMatch(opcode='LOP3', modifiers=('LUT',), operands=('P0', 'RZ', 'R8', 'R94', 'RZ', '0xfc', '!PT')),
        '@!P3 LDG.E.64.CONSTANT R6, desc[UR16][R26.64]': InstructionMatch(predicate='@!P3', opcode='LDG', modifiers=('E', '64', 'CONSTANT'), operands=('R6', 'desc[UR16][R26.64]')),
        '@!UP0 UIMAD UR7, UR10, 0xc, URZ': InstructionMatch(predicate='@!UP0', opcode='UIMAD', modifiers=(), operands=('UR7', 'UR10', '0xc', 'URZ')),
        '@UP0 LDCU.64 UR12, c[0x0][0x388]': InstructionMatch(predicate='@UP0', opcode='LDCU', modifiers=('64',), operands=('UR12', 'c[0x0][0x388]')),
        'RET.REL.NODEC R4 0x0': InstructionMatch(opcode='RET', modifiers=('REL', 'NODEC'), operands=('R4', '0x0')),
        '@!PT LDS RZ, [RZ]': InstructionMatch(predicate='@!PT', opcode='LDS', modifiers=(), operands=('RZ', '[RZ]')),
        'HMUL2 R0, R2.H0_H0, R3.H0_H0': InstructionMatch(opcode='HMUL2', modifiers=(), operands=('R0', 'R2.H0_H0', 'R3.H0_H0')),
        'UIADD3 UR5, UPT, UPT, -UR4, UR9, URZ': InstructionMatch(opcode='UIADD3', modifiers=(), operands=('UR5', 'UPT', 'UPT', '-UR4', 'UR9', 'URZ')),
        'BAR.SYNC.DEFER_BLOCKING 0x0': InstructionMatch(opcode='BAR', modifiers=('SYNC', 'DEFER_BLOCKING'), operands=('0x0',)),
        'STL [R1], R2': InstructionMatch(opcode='STL', modifiers=(), operands=('[R1]', 'R2')),
        'LDGSTS.E.LTC128B [R211+-0x4000], desc[UR24][R182.64+0x20]': InstructionMatch(opcode='LDGSTS', modifiers=('E', 'LTC128B'), operands=('[R211+-0x4000]', 'desc[UR24][R182.64+0x20]')),
        'STS.64 [R9.X8], R2': InstructionMatch(opcode='STS', modifiers=('64',), operands=('[R9.X8]', 'R2')),
        'LEA.HI.X R145, R140, R134.reuse, R135, 0x2, P3': InstructionMatch(opcode='LEA', modifiers=('HI', 'X'), operands=('R145', 'R140', 'R134.reuse', 'R135', '0x2', 'P3')),
    }
    """
    Zoo of real SASS instructions.
    """

    MATCHER: typing.Final[AnyMatcher] = AnyMatcher()

    def test_pattern(self) -> None:
        assert self.MATCHER.pattern.pattern == (
            r'(?:(?P<predicate>@!?U?P(?:T|[0-9]+))\s*)?'
            r'(?P<opcode>[A-Z0-9]+)'
            r'(?:\.(?P<modifiers>[A-Z0-9_]+))*\s*'
            r'(?:(?P<operands>[^,\s]+)(?:\s*,\s*|\s+))*'
            r'(?:(?P<operands>[^,\s]+))?'
        )

    @pytest.mark.parametrize(('instruction', 'expected'), INSTRUCTIONS.items())
    def test(self, instruction: str, expected: InstructionMatch) -> None:
        matched = self.MATCHER.match(inst=instruction)

        logging.info(f'{self.MATCHER} matched {instruction} as {matched}.')

        assert matched is not None
        assert matched == expected

    def test_no_match(self) -> None:
        assert self.MATCHER.match(inst='this-is-really-not-a-good-looking-instruction') is None
