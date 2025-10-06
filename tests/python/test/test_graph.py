import logging
import os
import pathlib

import pytest
import typeguard

from reprospect.tools.binaries import CuObjDump
from reprospect.tools.ncu      import Metric, Session, Report, ProfilingResults
from reprospect.tools.sass     import Decoder

from case import Case

class TestGraph(Case):
    """
    General test class.
    """
    TARGET_SOURCE = pathlib.Path('tests') / 'cpp' / 'cuda' / 'test_graph.cpp'

    DEMANGLED_NODE_A = {
        'NVIDIA' : 'void add_and_increment_kernel<(unsigned int)0, >(unsigned int *)',
        'Clang' : '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj',
    }

    class TestSASS:
        """
        `SASS`-focused analysis.
        """
        @pytest.fixture(scope = 'class')
        @classmethod
        def cubin(self, cwd, arch) -> pathlib.Path:
            return cwd / f'graph.1.{arch.as_sm}.cubin'

        @pytest.fixture(scope = 'class')
        @classmethod
        def cuobjdump(cls, target_file, arch, cwd, cubin) -> CuObjDump:
            return CuObjDump.extract(file = target_file, arch = arch, sass = True, cwd = cwd, cubin = cubin.name)[0]

        @typeguard.typechecked
        def test_function_count(self, cuobjdump : CuObjDump) -> None:
            """
            Count how many functions there are (1 per graph node).
            """
            assert len(cuobjdump.functions) == 4
            logging.info(cuobjdump.sass)

        @typeguard.typechecked
        def test_instruction_count(self, cuobjdump : CuObjDump) -> None:
            """
            Check how many instructions there are in the first graph node kernel.
            """
            decoder = Decoder(code = cuobjdump.functions[TestGraph.DEMANGLED_NODE_A[os.environ['CMAKE_CUDA_COMPILER_ID']]].code)
            assert len(decoder.instructions) >= 8

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
            assert len(results) == 4

        @typeguard.typechecked
        def test_launch_registers_per_thread_allocated_node_A(self, results : ProfilingResults) -> None:
            """
            Check metric `launch__registers_per_thread_allocated` for graph node A.
            """
            metrics = list(filter(lambda x: x['demangled'] == TestGraph.DEMANGLED_NODE_A[os.environ['CMAKE_CUDA_COMPILER_ID']], results.values()))
            assert len(metrics) == 1
            metrics = metrics[0]

            assert metrics[self.METRICS[0].name] == 512
