import logging
import os
import pathlib

import pytest
import typeguard

import reprospect

from reprospect.tools.binaries import CuObjDump
from reprospect.tools          import ncu
from reprospect.tools.sass     import Decoder

class TestGraph(reprospect.TestCase):
    """
    General test class.
    """
    TARGET_SOURCE = pathlib.Path('tests') / 'cpp' / 'cuda' / 'test_graph.cpp'

    DEMANGLED_NODE_A = {
        'NVIDIA' : 'void add_and_increment_kernel<(unsigned int)0, >(unsigned int *)',
        'Clang' : '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj',
    }

class TestSASS(TestGraph):
    """
    `SASS`-focused analysis.
    """
    @property
    @typeguard.typechecked
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'graph.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file = self.EXECUTABLE, arch = self.arch, sass = True, cwd = self.cwd, cubin = self.cubin.name)[0]

    @typeguard.typechecked
    def test_kernel_count(self, cuobjdump : CuObjDump) -> None:
        """
        Count how many kernels there are (1 per graph node).
        """
        assert len(cuobjdump.functions) == 4
        logging.info(cuobjdump.sass)

    @typeguard.typechecked
    def test_instruction_count(self, cuobjdump : CuObjDump) -> None:
        """
        Check how many instructions there are in the first graph node kernel.
        """
        decoder = Decoder(code = cuobjdump.functions[self.DEMANGLED_NODE_A[self.CMAKE_CUDA_COMPILER_ID]].code)
        assert len(decoder.instructions) >= 8

class TestNCU(TestGraph):
    """
    `ncu`-focused analysis.
    """

    METRICS = [
        ncu.Metric(name = 'launch__registers_per_thread_allocated')
    ]

    NVTX_INCLUDES = ['application-domain@outer-useless-range']

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def report(self) -> ncu.Report:
        session = ncu.Session(output = self.cwd / 'ncu')
        session.run(
            executable = self.EXECUTABLE,
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

    @typeguard.typechecked
    def test_launch_registers_per_thread_allocated_node_A(self, results : ncu.ProfilingResults) -> None:
        """
        Check metric `launch__registers_per_thread_allocated` for graph node A.
        """
        metrics = list(filter(lambda x: x['demangled'] == TestGraph.DEMANGLED_NODE_A[self.CMAKE_CUDA_COMPILER_ID], results.values()))
        assert len(metrics) == 1
        metrics = metrics[0]

        assert metrics[self.METRICS[0].name] == 512
