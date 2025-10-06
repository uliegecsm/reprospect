import pathlib

import pytest
import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.ncu          import Metric, Session, Report, ProfilingResults
from reprospect.tools.sass         import Decoder

from case import Case

class TestSaxpy(Case):
    """
    General test class.
    """
    TARGET_SOURCE = pathlib.Path('tests') / 'cpp' / 'cuda' / 'test_saxpy.cpp'

    class TestSASS:
        """
        `SASS`-focused analysis.
        """
        @pytest.fixture(scope = 'class')
        @classmethod
        def cubin(self, cwd, arch) -> pathlib.Path:
            return cwd / f'saxpy.1.{arch.as_sm}.cubin'

        @pytest.fixture(scope = 'class')
        @classmethod
        def cuobjdump(cls, target_file, arch : NVIDIAArch, cwd, cubin) -> CuObjDump:
            return CuObjDump.extract(file = target_file, arch = arch, sass = True, cwd = cwd, cubin = cubin.name)[0]

        @pytest.fixture(scope = 'class')
        @classmethod
        def decoder(cls, cuobjdump : CuObjDump) -> Decoder:
            return Decoder(code = cuobjdump.sass)

        @typeguard.typechecked
        def test_instruction_count(self, decoder : Decoder) -> None:
            assert len(decoder.instructions) >= 16

    class TestNCU:
        """
        `ncu`-focused analysis.
        """

        METRICS = [
            Metric(name = 'launch__registers_per_thread_allocated')
        ]

        NVTX_CAPTURE = 'application-domain@outer-useless-range'

        @pytest.fixture(scope = 'class')
        @classmethod
        @typeguard.typechecked
        def report(cls, target_file, cwd) -> Report:
            session = Session(output = cwd / 'ncu')
            session.run(
                cmd = [target_file],
                nvtx_capture = cls.NVTX_CAPTURE,
                cwd = cwd,
                metrics = cls.METRICS,
                retries = 5,
            )
            return Report(session = session)

        @pytest.fixture(scope = 'class')
        @classmethod
        @typeguard.typechecked
        def results(cls, report : Report) -> ProfilingResults:
            return report.extract_metrics_in_range(0, metrics = cls.METRICS)

        @typeguard.typechecked
        def test_result_count(self, report : Report, results : ProfilingResults) -> None:
            """
            Check how many ranges and results there are in the report.
            """
            assert report.report.num_ranges() == 1
            assert len(results) == 2
