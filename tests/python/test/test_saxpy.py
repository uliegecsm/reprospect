import pathlib

import pytest
import typeguard

import reprospect

from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.ncu          import Metric, Session, Report, ProfilingResults
from reprospect.tools.sass         import Decoder

class TestSaxpy(reprospect.TestCase):
    """
    General test class.
    """
    TARGET_SOURCE = pathlib.Path('tests') / 'cpp' / 'cuda' / 'test_saxpy.cpp'

class TestSASS(TestSaxpy):
    """
    `SASS`-focused analysis.
    """
    @property
    @typeguard.typechecked
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'saxpy.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file = self.EXECUTABLE, arch = self.arch, sass = True, cwd = self.cwd, cubin = self.cubin.name)[0]

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def decoder(self, cuobjdump : CuObjDump) -> Decoder:
        return Decoder(code = cuobjdump.sass)

    @typeguard.typechecked
    def test_instruction_count(self, decoder : Decoder) -> None:
        assert len(decoder.instructions) >= 16

class TestNCU(TestSaxpy):
    """
    `ncu`-focused analysis.
    """

    METRICS = [
        Metric(name = 'launch__registers_per_thread_allocated')
    ]

    NVTX_CAPTURE = 'application-domain@outer-useless-range'

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def report(self) -> Report:
        session = Session(output = self.cwd / 'ncu')
        session.run(
            executable = self.EXECUTABLE,
            nvtx_capture = self.NVTX_CAPTURE,
            cwd = self.cwd,
            metrics = self.METRICS,
            retries = 5,
        )
        return Report(session = session)

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def results(self, report : Report) -> ProfilingResults:
        return report.extract_metrics_in_range(0, metrics = self.METRICS)

    @typeguard.typechecked
    def test_result_count(self, report : Report, results : ProfilingResults) -> None:
        """
        Check how many ranges and results there are in the report.
        """
        assert report.report.num_ranges() == 1
        assert len(results) == 2
