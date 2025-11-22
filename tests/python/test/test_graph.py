import logging
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

class TestGraph(reprospect.CMakeAwareTestCase):
    """
    General test class.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'tests_cpp_cuda_graph'

    DEMANGLED_NODE_A : typing.Final[dict[str, str]] = {
        'NVIDIA' : 'void add_and_increment_kernel<(unsigned int)0, >(unsigned int *)',
        'Clang' : 'void add_and_increment_kernel<0u>(unsigned int*)',
    }

class TestSASS(TestGraph):
    """
    SASS-focused analysis.
    """
    @property
    def cubin(self) -> str:
        return f'graph.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope = 'class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(
            file = self.executable,
            arch = self.arch,
            sass = True,
            cwd = self.cwd,
            cubin = self.cubin,
            demangler = self.demangler,
        )[0]

    def test_kernel_count(self, cuobjdump : CuObjDump) -> None:
        """
        Count how many kernels there are (1 per graph node).
        """
        assert len(cuobjdump.functions) == 4
        logging.info(str(cuobjdump))

    def test_instruction_count(self, cuobjdump : CuObjDump) -> None:
        """
        Check how many instructions there are in the first graph node kernel.
        """
        decoder = Decoder(code = cuobjdump.functions[self.DEMANGLED_NODE_A[self.toolchains['CUDA']['compiler']['id']]].code)
        assert len(decoder.instructions) >= 8

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestNCU(TestGraph):
    """
    `ncu`-focused analysis.
    """

    METRICS : typing.Final[tuple[ncu.Metric]] = (
        ncu.Metric(name = 'launch__registers_per_thread_allocated'),
    )

    NVTX_INCLUDES : typing.Final[tuple[str]] = ('application_domain@outer_useless_range',)

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
        return report.extract_metrics_in_range(0, metrics = self.METRICS, demangler = self.demangler)

    def test_result_count(self, report : ncu.Report, results : ncu.ProfilingResults) -> None:
        """
        Check how many ranges and results there are in the report.
        """
        assert report.report.num_ranges() == 1
        assert len(results) == 4

    def test_launch_registers_per_thread_allocated_node_A(self, results : ncu.ProfilingResults) -> None:
        """
        Check metric `launch__registers_per_thread_allocated` for graph node A.
        """
        match self.toolchains['CUDA']['compiler']['id']:
            case 'NVIDIA':
                metrics_node_A = results.query_metrics(accessors = ('add_and_increment_kernel-0',))
            case 'Clang':
                NODE_A_MANGLED = '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj'
                metrics_node_A = results.query_metrics(accessors = (f'{NODE_A_MANGLED}-0',))
            case _:
                raise ValueError(f"unsupported compiler ID {self.toolchains['CUDA']['compiler']['id']}")

        assert metrics_node_A['demangled'] == self.DEMANGLED_NODE_A[self.toolchains['CUDA']['compiler']['id']]

        assert metrics_node_A['launch__registers_per_thread_allocated'] == 512
