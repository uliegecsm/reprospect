from reprospect.test.sass.matchers.move32 import Move32Matcher


class TestMove32Matcher:
    """
    Tests for :py:class:`reprospect.test.sass.matchers.move32.Move32Matcher`.
    """
    def test_mov(self) -> None:
        assert Move32Matcher().match(inst='MOV R11, 0x4') is not None
        assert Move32Matcher().match(inst='MOV R7, 0x3d53f941') is not None
        assert Move32Matcher().match(inst='MOV R5, c[0x0][0x0]') is not None

        assert Move32Matcher(dst='R42').match(inst='MOV R42, 0x1') is not None
        assert Move32Matcher(src='0x1').match(inst='MOV R42, 0x1') is not None

    def test_imad(self) -> None:
        assert (matched := Move32Matcher(src='R17').match(inst='IMAD.MOV.U32 R6, RZ, RZ, R17')) is not None
        assert matched.opcode == 'IMAD'
