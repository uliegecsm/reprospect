import pathlib

import pytest
import typeguard

import reprospect

from reprospect.tools.binaries import CuObjDump
from reprospect.tools          import ncu
from reprospect.tools.sass     import Decoder
from reprospect.utils          import detect

class TestSaxpy(reprospect.TestCase):
    """
    General test class.
    """
    NAME = 'tests_cpp_cuda_saxpy'

    TARGET_SOURCE = pathlib.Path('tests') / 'cpp' / 'cuda' / 'test_saxpy.cpp'

    @property
    @typeguard.typechecked
    def executable(self) -> pathlib.Path:
        return self.CMAKE_BINARY_DIR / self.TARGET_SOURCE.parent / self.NAME

class TestSASS(TestSaxpy):
    """
    SASS-focused analysis.
    """
    @property
    @typeguard.typechecked
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'saxpy.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file = self.executable, arch = self.arch, sass = True, cwd = self.cwd, cubin = self.cubin.name)[0]

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def decoder(self, cuobjdump : CuObjDump) -> Decoder:
        return Decoder(code = cuobjdump.sass)

    @typeguard.typechecked
    def test_instruction_count(self, decoder : Decoder) -> None:
        assert len(decoder.instructions) >= 16

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestNCU(TestSaxpy):
    """
    `ncu`-focused analysis.
    """

    METRICS = [
        ncu.Metric(name = 'launch__registers_per_thread_allocated')
    ]

    NVTX_INCLUDES = [
        'application_domain@launch_saxpy_kernel_first_time/',
        'application_domain@launch_saxpy_kernel_second_time/',
    ]

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def report(self) -> ncu.Report:
        session = ncu.Session(output = self.cwd / 'ncu')
        session.run(
            executable = self.executable,
            nvtx_includes = self.NVTX_INCLUDES,
            cwd = self.cwd,
            metrics = self.METRICS,
            retries = 5,
        )
        return ncu.Report(session = session)

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def results(self, report : ncu.Report) -> ncu.ProfilingResults:
        return report.extract_metrics_in_range(0, metrics = self.METRICS)

    @typeguard.typechecked
    def test_result_count(self, report : ncu.Report, results : ncu.ProfilingResults) -> None:
        """
        Check how many ranges and results there are in the report.
        """
        assert report.report.num_ranges() == 1
        assert len(results) == 4
