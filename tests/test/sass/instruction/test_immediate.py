import re
import typing

import pytest

from reprospect.test.sass.instruction import OpcodeModsWithOperandsMatcher
from reprospect.test.sass.instruction.immediate import Immediate
from reprospect.test.sass.instruction.pattern import PatternBuilder


class TestImmediate:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.immediate.Immediate`.
    """

    IMMEDIATES: typing.Final[tuple[str, ...]] = (
        '3.1400001049041748047', # '3.14f'
        '1.00000000000000000000e+10', # '1e10f'
        '0.5', # '.5f'
        '100', # '1.e2f'
        '-0.5', # '-.5f'
        '+QNAN', # '__int_as_float(0x7fc00000)',
        '0x7fffffff', # 'CUDART_NAN_F'
        '+INF', # 'CUDART_INF_F'
        '-QNAN', # '-CUDART_NAN_F'
        '-INF', # '-CUDART_INF_F'
    )
    """
    Zoo of real floating-point immediates.
    """

    def test_inf(self) -> None:
        assert re.fullmatch(Immediate.INF, '-INF') is not None
        assert re.fullmatch(Immediate.INF, '100') is None

        assert OpcodeModsWithOperandsMatcher(
            opcode='FADD', operands=(PatternBuilder.REG, PatternBuilder.REG, Immediate.INF),
        ).match('FADD R5, R0, +INF') is not None

    def test_qnan(self) -> None:
        assert re.fullmatch(Immediate.QNAN, '+QNAN') is not None
        assert re.fullmatch(Immediate.QNAN, '0x7fffffff') is not None
        assert re.fullmatch(Immediate.QNAN, '100') is None

        assert OpcodeModsWithOperandsMatcher(
            opcode='FADD', operands=(PatternBuilder.REG, PatternBuilder.REG, Immediate.QNAN),
        ).match('FADD R5, R0, -QNAN') is not None

    def test_floating(self) -> None:
        assert re.fullmatch(Immediate.FLOATING, '3.14') is not None
        assert re.fullmatch(Immediate.FLOATING, '+QNAN') is None

        assert OpcodeModsWithOperandsMatcher(
            opcode='FADD', operands=(PatternBuilder.REG, PatternBuilder.REG, Immediate.FLOATING),
        ).match('FADD R5, R0, 3.1400001049041748047') is not None

    def test_floating_or_limit(self) -> None:
        assert re.fullmatch(Immediate.FLOATING_OR_LIMIT, '0x3') is None

        assert OpcodeModsWithOperandsMatcher(
            opcode='FADD', operands=(PatternBuilder.REG, PatternBuilder.REG, Immediate.FLOATING_OR_LIMIT),
        ).match('FADD R5, R0, +QNAN') is not None

    @pytest.mark.parametrize('immediate', IMMEDIATES)
    def test_any(self, immediate: str) -> None:
        assert re.fullmatch(Immediate.FLOATING_OR_LIMIT, immediate) is not None
