import typing

import pytest

from reprospect.test.sass.instruction.constant import ConstantMatch, ConstantMatcher, OperandModifier


class TestConstantMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.constant.ConstantMatcher`.
    """
    CONSTANTS: typing.Final[dict[str, ConstantMatch]] = {
        'c[0x0][0x37c]': ConstantMatch(bank = '0x0', offset = '0x37c'),
        'c[0x0][R9]': ConstantMatch(bank = '0x0', offset = 'R9'),
        '-c[0x0][0x18c]': ConstantMatch(bank='0x0', offset='0x18c', modifier=OperandModifier.NEG),
        'c[0x0][UR456]': ConstantMatch(bank = '0x0', offset = 'UR456'),
    }

    MATCHER: typing.Final[ConstantMatcher] = ConstantMatcher()

    def test_constant(self) -> None:
        CONSTANT: typing.Final[str] = 'c[0x0][0x37c]'

        assert ConstantMatcher(bank='0x0'                  ).match(CONSTANT) == self.CONSTANTS[CONSTANT]
        assert ConstantMatcher(bank='0x0', offset = '0x37c').match(CONSTANT) == self.CONSTANTS[CONSTANT]
        assert ConstantMatcher(bank='0x2'                  ).match(CONSTANT) is None

    def test_constant_with_reg(self) -> None:
        CONSTANT: typing.Final[str] = 'c[0x0][R9]'

        assert ConstantMatcher(bank='0x0'               ).match(CONSTANT) == self.CONSTANTS[CONSTANT]
        assert ConstantMatcher(bank='0x0', offset = 'R9').match(CONSTANT) == self.CONSTANTS[CONSTANT]
        assert ConstantMatcher(bank='0x2'               ).match(CONSTANT) is None

    def test_constant_with_ureg(self) -> None:
        CONSTANT: typing.Final[str] = 'c[0x0][UR456]'

        assert ConstantMatcher(bank='0x0'                  ).match(CONSTANT) == self.CONSTANTS[CONSTANT]
        assert ConstantMatcher(bank='0x0', offset = 'UR456').match(CONSTANT) == self.CONSTANTS[CONSTANT]
        assert ConstantMatcher(bank='0x2'                  ).match(CONSTANT) is None

    def test_build_pattern(self) -> None:
        assert ConstantMatcher.build_pattern(bank='0x0', capture_bank=True, captured=False) == r'(?:(?P<modifier>[\-!\|~]))?c\[(?P<bank>0x0)\]\[(?:0x[0-9a-f]+|R[0-9]+|UR[0-9]+)\]'

    @pytest.mark.parametrize(('constant', 'expected'), CONSTANTS.items())
    def test_any(self, constant: str, expected: ConstantMatch) -> None:
        assert self.MATCHER.match(constant) == expected
