import typing

from reprospect.test.sass.instruction.register import RegisterMatch, RegisterType, RegisterMatcher

class TestRegisterMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.register.RegisterMatcher`.
    """
    def test_reg(self) -> None:
        REG : typing.Final[str] = 'R42'
        RES : typing.Final[RegisterMatch] = RegisterMatch(rtype = RegisterType.GPR, index = 42, reuse = False)

        assert RegisterMatcher().match(REG) == RES

        matcher = RegisterMatcher(rtype = RegisterType.GPR, special = False, reuse = False)
        assert matcher.pattern.pattern == r'(?P<rtype>(R))(?P<index>(\d+))'
        assert matcher.match(REG) == RES

    def test_regz(self) -> None:
        REG : typing.Final[str] = 'RZ'
        RES : typing.Final[RegisterMatch] = RegisterMatch(rtype = RegisterType.GPR, index = None, reuse = False)

        assert RegisterMatcher().match(REG) == RES

        matcher = RegisterMatcher(rtype = RegisterType.GPR, special = True)
        assert matcher.pattern.pattern == r'(?P<rtype>(R))(?P<special>(Z))'
        assert matcher.match(REG) == RES

    def test_ureg(self) -> None:
        REG : typing.Final[str] = 'UR42'
        RES : typing.Final[RegisterMatch] = RegisterMatch(rtype = RegisterType.UGPR, index = 42, reuse = False)

        assert RegisterMatcher().match(REG) == RES

        matcher = RegisterMatcher(rtype = RegisterType.UGPR, special = False)
        assert matcher.pattern.pattern == r'(?P<rtype>(UR))(?P<index>(\d+))(?P<reuse>\.reuse)?'
        assert matcher.match(REG) == RES

    def test_reg_reuse(self) -> None:
        assert RegisterMatcher().match('R42.reuse') == RegisterMatch(rtype = RegisterType.GPR, index = 42, reuse = True)

    def test_pred(self) -> None:
        assert RegisterMatcher().match('P3') == RegisterMatch(rtype = RegisterType.PRED, index = 3, reuse = False)

    def test_predt(self) -> None:
        REG : typing.Final[str] = 'PT'
        RES : typing.Final[RegisterMatch] = RegisterMatch(rtype = RegisterType.PRED, index = None, reuse = False)

        assert RegisterMatcher().match(REG) == RES

        matcher = RegisterMatcher(rtype = RegisterType.PRED)
        assert matcher.pattern.pattern == r'(?P<rtype>(P))(?P<special>(T))?(?P<index>(\d+))?'
        assert matcher.match(REG) == RES

    def test_upred(self) -> None:
        assert RegisterMatcher().match('UP3') == RegisterMatch(rtype = RegisterType.UPRED, index = 3, reuse = False)

    def test_upredt(self) -> None:
        assert RegisterMatcher().match('UPT') == RegisterMatch(rtype = RegisterType.UPRED, index = None, reuse = False)

    def test_not_a_reg(self) -> None:
        assert RegisterMatcher().match('YOLO666') is None
