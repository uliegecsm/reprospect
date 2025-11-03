import logging
import os
import re
import subprocess

import pytest
import semantic_version

from reprospect.tools.architecture import CUDA_SUPPORT, ComputeCapability, NVIDIAFamily, NVIDIAArch

class TestComputeCapability:
    """
    Tests for :py:class:`reprospect.tools.architecture.ComputeCapability`.
    """
    def test_from_int(self) -> None:
        assert ComputeCapability.from_int(86) == ComputeCapability(major = 8, minor = 6)

    def test_lt(self) -> None:
        cc = ComputeCapability(major = 8, minor = 6)

        assert cc >  70 and cc >  ComputeCapability(major = 7, minor = 0)
        assert cc >= 70 and cc >= ComputeCapability(major = 7, minor = 0)

        assert cc <  90 and cc <  ComputeCapability(major = 9, minor = 0)
        assert cc <= 90 and cc <= ComputeCapability(major = 9, minor = 0)

        assert cc == 86 and cc == ComputeCapability(major = 8, minor = 6)

    def test_supported(self) -> None:
        """
        Test :py:meth:`reprospect.tools.architecture.ComputeCapability.supported`.
        """
        # Use 'nvcc' to get the supported list of GPU code.
        supported = map(
            lambda x: ComputeCapability.from_int(int(re.match(r'sm_([0-9]+)', x).group(1))),
            subprocess.check_output(['nvcc', '--list-gpu-code']).decode().splitlines(),
        )
        CUDA_VERSION = semantic_version.Version(os.environ['CUDA_VERSION'])
        count = 0
        for cc in supported:
            if cc.as_int not in CUDA_SUPPORT:
                logging.info(f'{cc!r} is supported by CUDA {CUDA_VERSION} but is not listed in the CUDA support list.')
            else:
                logging.info(f'{cc!r} is supported by CUDA {CUDA_VERSION}.')
                assert CUDA_VERSION in CUDA_SUPPORT[cc.as_int]
                assert cc.supported(version = CUDA_VERSION)
                count += 1

        assert count > 0

        # VOLTA70 is supported until CUDA 13.
        assert not ComputeCapability.from_int(70).supported(version = semantic_version.Version('13.0.0'))

class TestNVIDIAFamily:
    """
    Tests for :py:class:`reprospect.tools.architecture.NVIDIAFamily`.
    """
    def test_from_compute_capability(self) -> None:
        assert NVIDIAFamily.from_compute_capability( 70) == NVIDIAFamily.VOLTA
        assert NVIDIAFamily.from_compute_capability( 86) == NVIDIAFamily.AMPERE
        assert NVIDIAFamily.from_compute_capability( 89) == NVIDIAFamily.ADA
        assert NVIDIAFamily.from_compute_capability(120) == NVIDIAFamily.BLACKWELL

    def test_to_string(self) -> None:
        assert str(NVIDIAFamily.VOLTA)  == 'VOLTA'
        assert str(NVIDIAFamily.AMPERE) == 'AMPERE'

    def test_from_string(self) -> None:
        assert NVIDIAFamily('VOLTA')  == NVIDIAFamily.VOLTA
        assert NVIDIAFamily('AMPERE') == NVIDIAFamily.AMPERE

class TestNVIDIAArch:
    """
    Tests for :py:class:`reprospect.tools.architecture.NVIDIAArch`.
    """
    def test_from_compute_capability(self) -> None:
        assert NVIDIAArch.from_compute_capability(cc = 70) == NVIDIAArch(family = NVIDIAFamily.VOLTA, compute_capability = ComputeCapability(major = 7, minor = 0))

    def test_from_str(self) -> None:
        assert NVIDIAArch.from_str('AMPERE86') == NVIDIAArch(family = NVIDIAFamily.AMPERE, compute_capability = ComputeCapability(major = 8, minor = 6))

        with pytest.raises(ValueError, match = 'unsupported architecture AMPERE86dd'):
            NVIDIAArch.from_str('AMPERE86dd')

    def test_as_compute(self) -> None:
        assert NVIDIAArch.from_str('HOPPER90').as_compute == 'compute_90'

    def test_as_sm(self) -> None:
        assert NVIDIAArch.from_compute_capability(cc = 80).as_sm == 'sm_80'

    def test_str(self) -> None:
        assert str(NVIDIAArch.from_compute_capability(75)) == 'TURING75'

    def test_repr(self) -> None:
        assert repr(NVIDIAArch.from_compute_capability(86)) == "NVIDIAArch(family=<NVIDIAFamily.AMPERE: 'AMPERE'>, compute_capability=ComputeCapability(major=8, minor=6))"

    def test_str_cycle(self) -> None:
        arch = NVIDIAArch.from_compute_capability(70)
        assert NVIDIAArch.from_str(str(arch)) == arch
