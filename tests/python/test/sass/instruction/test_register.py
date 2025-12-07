import pytest

from reprospect.test.sass.instruction import RegisterMatch
from reprospect.tools.sass.decode     import RegisterType

class TestRegisterMatch:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.register.RegisterMatch`.
    """
    def test_reg(self) -> None:
        assert RegisterMatch.parse('R42') == RegisterMatch(rtype = RegisterType.GPR, index = 42, reuse = False)

    def test_regz(self) -> None:
        assert RegisterMatch.parse('RZ') == RegisterMatch(rtype = RegisterType.GPR, index = -1, reuse = False)

    def test_ureg(self) -> None:
        assert RegisterMatch.parse('UR42') == RegisterMatch(rtype = RegisterType.UGPR, index = 42, reuse = False)

    def test_reg_reuse(self) -> None:
        assert RegisterMatch.parse('R42.reuse') == RegisterMatch(rtype = RegisterType.GPR, index = 42, reuse = True)

    def test_pred(self) -> None:
        assert RegisterMatch.parse('P3') == RegisterMatch(rtype = RegisterType.PRED, index = 3, reuse = False)

    def test_predt(self) -> None:
        assert RegisterMatch.parse('PT') == RegisterMatch(rtype = RegisterType.PRED, index = -1, reuse = False)

    def test_upred(self) -> None:
        assert RegisterMatch.parse('UP3') == RegisterMatch(rtype = RegisterType.UPRED, index = 3, reuse = False)

    def test_upredt(self) -> None:
        assert RegisterMatch.parse('UPT') == RegisterMatch(rtype = RegisterType.UPRED, index = -1, reuse = False)

    def test_not_a_reg(self) -> None:
        with pytest.raises(ValueError, match = "Invalid register format 'YOLO666'."):
            RegisterMatch.parse('YOLO666')
