from reprospect.test.sass.instruction.integer import (
    IntAdd3Matcher,
    IntAddMatcher,
    LEAMatcher,
)
from reprospect.tools.architecture import NVIDIAArch


class TestIntAddMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.integer.IntAddMatcher`.
    """
    def test(self) -> None:
        matcher = IntAddMatcher(dst='R14', src_a='R45')

        assert matcher.match(inst='IADD R14, R45, R11') is not None
        assert matcher.match(inst='IADD R15, R45, R11') is None

class TestIntAdd3Matcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.integer.IntAdd3Matcher`.
    """
    def test_before_blackwell(self) -> None:
        matcher = IntAdd3Matcher(src_c='R66', arch=NVIDIAArch.from_compute_capability(90))

        assert matcher.match(inst='IADD3 R17, R14, R11, R66') is not None
        assert matcher.match(inst='IADD3 R17, R14, R11, R65') is None

    def test_as_of_blackwell(self) -> None:
        matcher = IntAdd3Matcher(src_c='R66', arch=NVIDIAArch.from_compute_capability(120))

        assert matcher.match(inst='IADD3 R17, PT, PT, R14, R11, R66') is not None
        assert matcher.match(inst='IADD3 R17, PT, PT, R14, R11, R65') is None

class TestLEAMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.integer.LEAMatcher`.
    """
    def test(self) -> None:
        matcher = LEAMatcher()

        assert (matched := matcher.match(inst='LEA R9, R2, R5, 0x1')) is not None
        assert matched.opcode == 'LEA'
        assert matched.operands == ('R9', 'R2', 'R5', '0x1')
        assert matched.additional is not None
        assert matched.additional['shift'] == ['0x1']

    def test_shift(self) -> None:
        """
        Enforce the shift.
        """
        matcher = LEAMatcher(shift='0x3')

        assert matcher.match(inst='LEA R9, R2, R5, 0x1') is None

        assert (matched := matcher.match(inst='LEA R9, R2, R5, 0x3')) is not None

        assert matched.additional is not None
        assert matched.additional['shift'] == ['0x3']
