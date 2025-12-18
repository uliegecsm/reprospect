import logging
import pathlib
import sys
import typing

import pandas
import pytest

from reprospect.test  import CMakeAwareTestCase, environment
from reprospect.tools import nsys
from reprospect.utils import detect

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class TestDispatch(CMakeAwareTestCase):
    """
    Trace the CUDA API calls during :code:`Kokkos::Experimental::Graph` stages.

    It uses :file:`examples/kokkos/graph/example_dispatch.cpp`.
    """
    KOKKOS_TOOLS_NVTX_CONNECTOR_LIB = environment.EnvironmentField(converter = pathlib.Path)
    """Used in :py:meth:`TestNSYS.report`."""

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_graph_dispatch'

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestNSYS(TestDispatch):
    """
    `nsys`-focused analysis.
    """
    NODE_COUNT: typing.Final[int] = 5

    @pytest.fixture(scope = 'class')
    def report(self) -> nsys.Report:
        """
        Analyse with `nsys`, use :py:class:`reprospect.tools.nsys.Cacher`.
        """
        with nsys.Cacher() as cacher:
            command = nsys.Command(
                executable = self.executable,
                output = self.cwd / self.executable.name,
                opts = ('--cuda-graph-trace=node',),
                nvtx_capture = 'dispatch',
                args = (
                    f'--kokkos-tools-libs={self.KOKKOS_TOOLS_NVTX_CONNECTOR_LIB}',
                ),
            )
            entry = cacher.run(command = command, cwd = self.cwd)

            return nsys.Report(db = cacher.export_to_sqlite(command = command, entry = entry))

    @staticmethod
    def get(*, report: nsys.Report, kernels: pandas.DataFrame, label: str) -> pandas.DataFrame:
        """
        Get kernels from `kernels` table that are correlated to the :code:`cudaGraphLaunch` API call in the NVTX region `label`.
        """
        api = report.get_events(table = 'CUPTI_ACTIVITY_KIND_RUNTIME', accessors = ('dispatch', label))

        launch = nsys.Report.single_row(data = api[api['name'].apply(nsys.strip_cuda_api_suffix) == 'cudaGraphLaunch'])

        return report.get_correlated_rows(src = launch, dst = kernels)

    def test_streams(self, report: nsys.Report) -> None:
        """
        Each kernel gets a unique stream ID.

        It means that at the CUDA backend level, all nodes are shown to the kernel scheduler
        as independent and may be executed concurrently.

        It must be noted that CUDA does not provide a way to create a graph node and enforce the stream on which it will eventually run.
        This has motivated a refactoring of the :code:`Kokkos::Experimental::Graph` API, see https://github.com/kokkos/kokkos/pull/8191.
        """
        with report:
            # A bit of debug log.
            logging.info(report.nvtx_events)

            # Find the stream creation call.
            # Note that it is not possible to correlate the call with the stream ID it created.
            api = report.get_events(table = 'CUPTI_ACTIVITY_KIND_RUNTIME', accessors = ('dispatch', 'create stream'))
            create = nsys.Report.single_row(data = api[api['name'].apply(nsys.strip_cuda_api_suffix) == 'cudaStreamCreate'])
            logging.info(f'CUDA stream create API call:\n{create}')

            # However, the stream is used for the fencing, so its stream ID can still be retrieved.
            api = report.get_events(table = 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', accessors = ('dispatch', 'after graph submissions'), stringids = None)
            fence = nsys.Report.single_row(data = api)
            logging.info(f'CUDA stream synchronize API call:\n{fence}')

            ENUM_CUPTI_SYNC_TYPE = report.table(name = 'ENUM_CUPTI_SYNC_TYPE')
            CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT = report.single_row(data = ENUM_CUPTI_SYNC_TYPE[ENUM_CUPTI_SYNC_TYPE['name'] == 'CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT'])

            assert fence['syncType'] == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT['id']

            stream_id = fence['streamId']

            logging.info(f'The create stream ID is {stream_id}.')

            # Get all kernels.
            kernels = report.table(name = 'CUPTI_ACTIVITY_KIND_KERNEL')

            # Select kernels from the first graph submission.
            kernels_0 = self.get(report = report, kernels = kernels, label = 'graph - submit - 0')
            assert len(kernels_0) == self.NODE_COUNT

            streams_0 = kernels_0['streamId'].to_list()

            logging.info(f'Kernels for the first submission ran on {streams_0}.')

            assert len(streams_0) == len(set(streams_0))

            # Select kernels from the second graph submission.
            # Second submission reuses the streams of the first submission.
            kernels_1 = self.get(report = report, kernels = kernels, label = 'graph - submit - 1')
            assert len(kernels_1) == self.NODE_COUNT

            streams_1 = kernels_1['streamId'].to_list()

            logging.info(f'Kernels for the second submission ran on {streams_1}.')

            assert sorted(streams_0) == sorted(streams_1)

            # The first node actually got the stream ID of the stream passed to the graph launch API call.
            assert stream_id in streams_0
