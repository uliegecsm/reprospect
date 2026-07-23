"""
Check the CUDA API calls made when defining a :code:`Kokkos::Experimental::Graph`.
"""

import logging
import pathlib
import sys

import pytest

from reprospect.testing import CMakeAwareTestCase, environment
from reprospect.tools import nsys
from reprospect.utils import detect
from reprospect.utils.rich_helpers import df_to_table, to_string

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class TestDefinition(CMakeAwareTestCase):
    """
    Trace the CUDA API calls during :code:`Kokkos::Experimental::Graph` definition stage.

    It uses :file:`examples/kokkos/graph/example_definition.cpp`.
    """
    KOKKOS_TOOLS_NVTX_CONNECTOR_LIB = environment.EnvironmentField(converter=pathlib.Path)
    """Used in :py:meth:`TestNSYS.report`."""

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_graph_definition'

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestNSYS(TestDefinition):
    """
    `nsys`-focused analysis.
    """
    @pytest.fixture(scope='class')
    def report(self) -> nsys.Report:
        """
        Analyse with `nsys`, use :py:class:`reprospect.tools.nsys.Cacher`.
        """
        with nsys.Cacher() as cacher:
            command = nsys.Command(
                executable=self.executable,
                output=self.cwd / self.executable.name,
                opts=('--cuda-graph-trace=node',),
                nvtx_capture='definition',
                args=(
                    f'--kokkos-tools-libs={self.KOKKOS_TOOLS_NVTX_CONNECTOR_LIB}',
                ),
            )
            entry = cacher.run(command=command, cwd=self.cwd)

            return nsys.Report(db=cacher.export_to_sqlite(command=command, entry=entry))

    def test_api_calls(self, report: nsys.Report) -> None:
        """
        Check the sequence of CUDA API calls to create the graph, nodes and edges.
        """
        with report:
            logging.info(report.nvtx_events)

            api = report.get_events(table='CUPTI_ACTIVITY_KIND_RUNTIME', accessors=('definition', 'graph - definition'))

            logging.info(to_string(df_to_table(api)))

            expt = (
                'cudaGraphCreate',
                # The root node is created.
                'cudaGraphAddEmptyNode',
                # Node A is added.
                'cudaGraphAddKernelNode',
                'cudaGraphAddDependencies',
                # Node B is added.
                'cudaGraphAddKernelNode',
                'cudaGraphAddDependencies',
                # Node C is added.
                'cudaGraphAddKernelNode',
                'cudaGraphAddDependencies',
                # The aggregate node is added.
                'cudaGraphAddEmptyNode',
                'cudaGraphAddDependencies',
                'cudaGraphAddDependencies',
                # Node D is added.
                'cudaGraphAddKernelNode',
                'cudaGraphAddDependencies',
            )

            expt_it = iter(expt)
            current = next(expt_it, None)

            for name in map(nsys.strip_cuda_api_suffix, api['name']):
                if name == current:
                    current = next(expt_it, None)
                    if current is None:
                        break

            assert current is None
