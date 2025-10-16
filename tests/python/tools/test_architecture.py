from reprospect.tools.architecture import ComputeCapability, NVIDIAFamily, NVIDIAArch

class TestComputeCapability:
    """
    Tests for :py:class:`reprospect.tools.architecture.ComputeCapability`.
    """
    def test_from_int(self):
        assert ComputeCapability.from_int(86) == ComputeCapability(major = 8, minor = 6)

    def test_lt(self):
        cc = ComputeCapability(major = 8, minor = 6)

        assert cc >  70 and cc >  ComputeCapability(major = 7, minor = 0)
        assert cc >= 70 and cc >= ComputeCapability(major = 7, minor = 0)

        assert cc <  90 and cc <  ComputeCapability(major = 9, minor = 0)
        assert cc <= 90 and cc <= ComputeCapability(major = 9, minor = 0)

        assert cc == 86 and cc == ComputeCapability(major = 8, minor = 6)

class TestNVIDIAFamily:
    """
    Tests for :py:class:`reprospect.tools.architecture.NVIDIAFamily`.
    """
    def test_from_compute_capability(self):
        assert NVIDIAFamily.from_compute_capability( 70) == NVIDIAFamily.VOLTA
        assert NVIDIAFamily.from_compute_capability( 86) == NVIDIAFamily.AMPERE
        assert NVIDIAFamily.from_compute_capability( 89) == NVIDIAFamily.ADA
        assert NVIDIAFamily.from_compute_capability(120) == NVIDIAFamily.BLACKWELL

    def test_to_string(self):
        assert str(NVIDIAFamily.VOLTA)  == 'VOLTA'
        assert str(NVIDIAFamily.AMPERE) == 'AMPERE'

    def test_from_string(self):
        assert NVIDIAFamily('VOLTA')  == NVIDIAFamily.VOLTA
        assert NVIDIAFamily('AMPERE') == NVIDIAFamily.AMPERE

class TestNVIDIAArch:
    """
    Tests for :py:class:`reprospect.tools.architecture.NVIDIAArch`.
    """
    def test_from_compute_capability(self):
        assert NVIDIAArch.from_compute_capability(cc = 70) == NVIDIAArch(family = NVIDIAFamily.VOLTA, compute_capability = ComputeCapability(major = 7, minor = 0))

    def test_from_str(self):
        assert NVIDIAArch.from_str('AMPERE86') == NVIDIAArch(family = NVIDIAFamily.AMPERE, compute_capability = ComputeCapability(major = 8, minor = 6))

    def test_as_compute(self):
        assert NVIDIAArch.from_str('HOPPER90').as_compute == 'compute_90'

    def test_as_sm(self):
        assert NVIDIAArch.from_compute_capability(cc = 80).as_sm == 'sm_80'

    def test_str(self):
        assert str(NVIDIAArch.from_compute_capability(75)) == 'TURING75'

    def test_repr(self):
        assert repr(NVIDIAArch.from_compute_capability(86)) == "NVIDIAArch(family=<NVIDIAFamily.AMPERE: 'AMPERE'>, compute_capability=ComputeCapability(major=8, minor=6))"

    def test_str_cycle(self):
        arch = NVIDIAArch.from_compute_capability(70)
        assert NVIDIAArch.from_str(str(arch)) == arch
