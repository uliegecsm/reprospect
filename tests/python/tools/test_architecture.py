from cuda_helpers.tools.architecture import NVIDIAFamily, NVIDIAArch

class TestNVIDIAFamily:

    def test_from_compute_capability(self):
        assert NVIDIAFamily.from_compute_capability('70') == NVIDIAFamily.VOLTA
        assert NVIDIAFamily.from_compute_capability( 86 ) == NVIDIAFamily.AMPERE
        assert NVIDIAFamily.from_compute_capability( 89 ) == NVIDIAFamily.ADA
        assert NVIDIAFamily.from_compute_capability( 120) == NVIDIAFamily.BLACKWELL

    def test_to_string(self):
        assert str(NVIDIAFamily.VOLTA)  == 'VOLTA'
        assert str(NVIDIAFamily.AMPERE) == 'AMPERE'

    def test_from_string(self):
        assert NVIDIAFamily('VOLTA')  == NVIDIAFamily.VOLTA
        assert NVIDIAFamily('AMPERE') == NVIDIAFamily.AMPERE

class TestNVIDIAArch:

    def test_from_compute_capability(self):
        assert NVIDIAArch.from_compute_capability(cc = 70) == NVIDIAArch(family = NVIDIAFamily.VOLTA, compute_capability = 70)

    def test_from_str(self):
        assert NVIDIAArch.from_str('AMPERE86') == NVIDIAArch(family = NVIDIAFamily.AMPERE, compute_capability = 86)

    def test_as_compute(self):
        assert NVIDIAArch.from_str('HOPPER90').as_compute == 'compute_90'

    def test_as_sm(self):
        assert NVIDIAArch.from_compute_capability(cc = 80).as_sm == 'sm_80'

    def test_str(self):
        assert str(NVIDIAArch.from_compute_capability(75)) == 'TURING75'

    def test_repr(self):
        assert repr(NVIDIAArch.from_compute_capability(86)) == "NVIDIAArch(family=<NVIDIAFamily.AMPERE: 'AMPERE'>, compute_capability=86)"

    def test_str_cycle(self):
        arch = NVIDIAArch.from_compute_capability(70)
        assert NVIDIAArch.from_str(str(arch)) == arch
