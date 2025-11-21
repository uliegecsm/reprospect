import pathlib
import sys
import typing

import pytest

import reprospect

from reprospect.tools.binaries import CuObjDump
from reprospect.tools          import ncu
from reprospect.tools.sass     import Decoder
from reprospect.utils          import detect

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class TestSaxpy(reprospect.CMakeAwareTestCase):
    """
    General test class.
    """
    NAME : typing.Final[str] = 'tests_cpp_cuda_saxpy'

    TARGET_SOURCE : typing.Final[pathlib.Path] = pathlib.Path('tests') / 'cpp' / 'cuda' / 'test_saxpy.cpp'

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'tests_cpp_cuda_saxpy'

class TestSASS(TestSaxpy):
    """
    SASS-focused analysis.
    """
    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'saxpy.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope = 'class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file = self.executable, arch = self.arch, sass = True, cwd = self.cwd, cubin = self.cubin.name)[0]

    @pytest.fixture(scope = 'class')
    def decoder(self, cuobjdump : CuObjDump) -> Decoder:
        assert len(cuobjdump.functions) == 1

        return Decoder(code = next(iter(cuobjdump.functions.values())).code)

    def test_instruction_count(self, decoder : Decoder) -> None:
        assert len(decoder.instructions) >= 16

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestNCU(TestSaxpy):
    """
    `ncu`-focused analysis.
    """

    METRICS : typing.Final[tuple[ncu.Metric]] = (
        ncu.Metric(name = 'launch__registers_per_thread_allocated'),
    )

    NVTX_INCLUDES : typing.Final[tuple[str, ...]] = (
        'application_domain@launch_saxpy_kernel_first_time/',
        'application_domain@launch_saxpy_kernel_second_time/',
    )

    @pytest.fixture(scope = 'class')
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
    def results(self, report : ncu.Report) -> ncu.ProfilingResults:
        return report.extract_metrics_in_range(0, metrics = self.METRICS)

    def test_result_count(self, report : ncu.Report, results : ncu.ProfilingResults) -> None:
        """
        Check how many ranges and results there are in the report.
        """
        assert report.report.num_ranges() == 1
        assert len(results) == 4
