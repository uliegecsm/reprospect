import typing

from reprospect.test.sass.instruction.constant import ConstantMatch, ConstantMatcher


class TestConstantMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.constant.ConstantMatcher`.
    """
    def test_constant(self) -> None:
        CONSTANT: typing.Final[str] = 'c[0x0][0x37c]'

        assert ConstantMatcher(                          ).match(CONSTANT) == ConstantMatch(bank='0x0', offset='0x37c')
        assert ConstantMatcher(bank='0x0'                ).match(CONSTANT) == ConstantMatch(bank='0x0', offset='0x37c')
        assert ConstantMatcher(bank='0x0', offset='0x37c').match(CONSTANT) == ConstantMatch(bank='0x0', offset='0x37c')
        assert ConstantMatcher(bank='0x2'                ).match(CONSTANT) is None

    def test_constant_with_reg(self) -> None:
        CONSTANT: typing.Final[str] = 'c[0x0][R9]'

        assert ConstantMatcher(                       ).match(CONSTANT) == ConstantMatch(bank='0x0', offset='R9')
        assert ConstantMatcher(bank='0x0'             ).match(CONSTANT) == ConstantMatch(bank='0x0', offset='R9')
        assert ConstantMatcher(bank='0x0', offset='R9').match(CONSTANT) == ConstantMatch(bank='0x0', offset='R9')
        assert ConstantMatcher(bank='0x2'             ).match(CONSTANT) is None

    def test_constant_with_ureg(self) -> None:
        CONSTANT: typing.Final[str] = 'c[0x0][UR456]'

        assert ConstantMatcher(                          ).match(CONSTANT) == ConstantMatch(bank='0x0', offset='UR456')
        assert ConstantMatcher(bank='0x0'                ).match(CONSTANT) == ConstantMatch(bank='0x0', offset='UR456')
        assert ConstantMatcher(bank='0x0', offset='UR456').match(CONSTANT) == ConstantMatch(bank='0x0', offset='UR456')
        assert ConstantMatcher(bank='0x2'                ).match(CONSTANT) is None

    def test_build_pattern(self) -> None:
        assert ConstantMatcher.build_pattern(bank='0x0', capture_bank=True, captured=False) == r'c\[(?P<bank>0x0)\]\[(?:0x[0-9a-f]+|R[0-9]+|UR[0-9]+)\]'
