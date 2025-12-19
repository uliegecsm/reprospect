import typing

import pytest

from reprospect.test.sass.instruction.register import (
    MathModifier,
    RegisterMatch,
    RegisterMatcher,
    RegisterType,
)


class TestRegisterMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.register.RegisterMatcher`.
    """
    REGISTERS: typing.Final[dict[str, RegisterMatch]] = {
        'R42':       RegisterMatch(rtype=RegisterType.GPR,   index=42,   reuse=False),
        '!P0':       RegisterMatch(rtype=RegisterType.PRED,  index=0,    reuse=False, math=MathModifier.NOT),
        '!R42':      RegisterMatch(rtype=RegisterType.GPR,   index=42,   reuse=False, math=MathModifier.NOT),
        '~R42':      RegisterMatch(rtype=RegisterType.GPR,   index=42,   reuse=False, math=MathModifier.INV),
        '-R42':      RegisterMatch(rtype=RegisterType.GPR,   index=42,   reuse=False, math=MathModifier.NEG),
        '|R42|':     RegisterMatch(rtype=RegisterType.GPR,   index=42,   reuse=False, math=MathModifier.ABS),
        'R42.reuse': RegisterMatch(rtype=RegisterType.GPR,   index=42,   reuse=True),
        'RZ':        RegisterMatch(rtype=RegisterType.GPR,   index=None, reuse=False),
        'UR42':      RegisterMatch(rtype=RegisterType.UGPR,  index=42,   reuse=False),
        'P3':        RegisterMatch(rtype=RegisterType.PRED,  index=3,    reuse=False),
        'PT':        RegisterMatch(rtype=RegisterType.PRED,  index=None, reuse=False),
        'UP3':       RegisterMatch(rtype=RegisterType.UPRED, index=3,    reuse=False),
        'UPT':       RegisterMatch(rtype=RegisterType.UPRED, index=None, reuse=False),
    }

    MATCHER: typing.Final[RegisterMatcher] = RegisterMatcher()

    def test_reg(self) -> None:
        REG: typing.Final[str] = 'R42'

        matcher = RegisterMatcher(rtype=RegisterType.GPR, special=False, reuse=False)
        assert matcher.pattern.pattern == r'(?:(?P<modifier_math>[\-!\|~]))?(?P<rtype>R)(?P<index>\d+)'
        assert matcher.match(REG) == self.REGISTERS[REG]

    def test_regz(self) -> None:
        REG: typing.Final[str] = 'RZ'

        matcher = RegisterMatcher(rtype=RegisterType.GPR, special=True)
        assert matcher.pattern.pattern == r'(?:(?P<modifier_math>[\-!\|~]))?(?P<rtype>R)(?P<special>Z)'
        assert matcher.match(REG) == self.REGISTERS[REG]

    def test_ureg(self) -> None:
        REG: typing.Final[str] = 'UR42'

        matcher = RegisterMatcher(rtype=RegisterType.UGPR, special=False)
        assert matcher.pattern.pattern == r'(?:(?P<modifier_math>[\-!\|~]))?(?P<rtype>UR)(?P<index>\d+)(?P<reuse>\.reuse)?'
        assert matcher.match(REG) == self.REGISTERS[REG]

    def test_reg_reuse(self) -> None:
        assert self.MATCHER.match('R42.reuse') == self.REGISTERS['R42.reuse']

    def test_pred(self) -> None:
        assert self.MATCHER.match('P3') == self.REGISTERS['P3']

    def test_predt(self) -> None:
        REG: typing.Final[str] = 'PT'

        matcher = RegisterMatcher(rtype=RegisterType.PRED)
        assert matcher.pattern.pattern == r'(?:(?P<modifier_math>[\-!\|~]))?(?P<rtype>P)(?P<special>T)?(?P<index>\d+)?'
        assert matcher.match(REG) == self.REGISTERS[REG]

    def test_upred(self) -> None:
        assert self.MATCHER.match('UP3') == self.REGISTERS['UP3']

    def test_upredt(self) -> None:
        assert self.MATCHER.match('UPT') == self.REGISTERS['UPT']

    def test_pred_not(self) -> None:
        matcher = RegisterMatcher(rtype=RegisterType.PRED, math=MathModifier.NOT)
        assert matcher.match('!P0') == self.REGISTERS['!P0']

    def test_not_a_reg(self) -> None:
        assert self.MATCHER.match('YOLO666') is None

    @pytest.mark.parametrize(('register', 'expected'), REGISTERS.items())
    def test_any(self, register, expected) -> None:
        assert self.MATCHER.match(reg=register) == expected
