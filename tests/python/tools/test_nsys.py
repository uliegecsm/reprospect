import logging
import os
import pathlib
import sys
import tempfile
import typing

import pytest
import semantic_version
import typeguard

from reprospect.tools.nsys import Report, Session, strip_cuda_api_suffix, Cacher
from reprospect.utils      import detect

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestSession:
    """
    Test :py:class:`reprospect.tools.nsys.Session`.
    """
    EXECUTABLE = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy' if 'CMAKE_BINARY_DIR' in os.environ else None

    @typeguard.typechecked
    def run(self, nvtx_capture : typing.Optional[str] = None) -> Session:
        assert self.EXECUTABLE.is_file()

        ns = Session(
            output_dir = TMPDIR,
            output_file_prefix = self.EXECUTABLE.name,
        )

        ns.run(
            executable = self.EXECUTABLE,
            nvtx_capture = nvtx_capture,
            cwd = TMPDIR,
        )

        ns.export_to_sqlite()

        return ns

    def test_cuda_api_trace(self):
        """
        Collect all `Cuda` API calls of :file:`tests/cpp/cuda/test_saxpy.cpp`.
        """
        ns = self.run(nvtx_capture = "outer_useless_range@application_domain")

        cuda_api_trace = ns.extract_statistical_report(report = 'cuda_api_trace')

        logging.info(cuda_api_trace['Name'])

        expt = [
            'cuModuleGetLoadingMode',
            'cudaStreamCreate',
            'cudaStreamCreate',
            'cudaMallocAsync',
            'cudaMallocAsync',
            'cudaMemcpyAsync',
            'cudaMemcpyAsync',
            'cudaStreamSynchronize',
        ]

        if semantic_version.Version(os.environ['CUDA_VERSION']) in semantic_version.SimpleSpec('>=13.0.0'):
            expt += [
                'cuLibraryLoadData',
                'cuLibraryGetKernel',
                'cuKernelGetName',
            ]

        expt += ['cudaLaunchKernel']

        if semantic_version.Version(os.environ['CUDA_VERSION']) in semantic_version.SimpleSpec('>=13.0.0'):
            expt += ['cuKernelGetName']

        expt += [
            'cudaLaunchKernel',
            'cudaMemcpyAsync',
            'cudaStreamSynchronize',
            'cudaFreeAsync',
            'cudaFreeAsync',
            'cudaStreamDestroy',
            'cudaStreamDestroy',
        ]

        assert list(map(strip_cuda_api_suffix, cuda_api_trace['Name'].to_list())) == expt

    def test_report(self):
        """
        Process the `nsys` report with :py:class:`reprospect.tools.nsys.Report`.
        """
        ns = self.run()

        cuda_api_trace = ns.extract_statistical_report(report = 'cuda_api_trace')

        with Report(db = ns.output_file_sqlite) as report:

            logging.info(f'Tables are {report.tables}.')

            cupti_activity_kind_synchronization = report.table(name = 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')

            logging.info(cupti_activity_kind_synchronization)

            cuda_stream_synchronize = cuda_api_trace[cuda_api_trace['Name'].str.startswith('cudaStreamSynchronize')]

            logging.info(cuda_stream_synchronize)

            assert len(cuda_stream_synchronize) == 2

            # Each call to 'cudaStreamSynchronize' targets a distinct stream.
            stream_id_A = report.get_correlated_row(src = cuda_stream_synchronize.iloc[0], dst = cupti_activity_kind_synchronization, correlation_src = 'CorrID')['streamId']
            stream_id_B = report.get_correlated_row(src = cuda_stream_synchronize.iloc[1], dst = cupti_activity_kind_synchronization, correlation_src = 'CorrID')['streamId']

            assert stream_id_A != stream_id_B

            # Check that the 'saxpy' kernels ran on stream B.
            cupti_activity_kind_kernel = report.table(name = 'CUPTI_ACTIVITY_KIND_KERNEL')
            saxpy_kernel_first  = cupti_activity_kind_kernel.iloc[0]
            saxpy_kernel_second = cupti_activity_kind_kernel.iloc[1]

            assert saxpy_kernel_first ['streamId'] == stream_id_B
            assert saxpy_kernel_second['streamId'] == stream_id_B

            # Check 'saxpy' kernels launch block/grid configuration.
            for kernel in [saxpy_kernel_first, saxpy_kernel_second]:
                assert kernel['gridX'] == 8
                assert kernel['gridY'] == 1
                assert kernel['gridZ'] == 1

                assert kernel['blockX'] == 128
                assert kernel['blockY'] == 1
                assert kernel['blockZ'] == 1

            # Check 'saxpy' kernels launch type.
            ENUM_CUDA_KERNEL_LAUNCH_TYPE = report.table(name = 'ENUM_CUDA_KERNEL_LAUNCH_TYPE')

            CUDA_KERNEL_LAUNCH_TYPE_REGULAR = report.single_row(data = ENUM_CUDA_KERNEL_LAUNCH_TYPE[ENUM_CUDA_KERNEL_LAUNCH_TYPE['name'] == 'CUDA_KERNEL_LAUNCH_TYPE_REGULAR'])['id']
            assert saxpy_kernel_first ['launchType'] == CUDA_KERNEL_LAUNCH_TYPE_REGULAR
            assert saxpy_kernel_second['launchType'] == CUDA_KERNEL_LAUNCH_TYPE_REGULAR

            # Check 'saxpy' kernels mangled and demangled names.
            stringids = report.table(name = 'StringIds')
            for kernel in [saxpy_kernel_first, saxpy_kernel_second]:
                assert report.single_row(data = stringids[stringids['id'] == kernel['mangledName'  ]])['value'] == '_Z12saxpy_kerneljfPKfPf'
                assert report.single_row(data = stringids[stringids['id'] == kernel['demangledName']])['value'] == 'saxpy_kernel(unsigned int, float, const float *, float *)'

class TestCacher:
    """
    Tests for :py:class:`reprospect.tools.nsys.Cacher`.
    """
    GRAPH = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_graph' if 'CMAKE_BINARY_DIR' in os.environ else None
    SAXPY = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy' if 'CMAKE_BINARY_DIR' in os.environ else None

    def test_hash_same(self):
        """
        Test :py:meth:`reprospect.tools.nsys.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir, session = Session(output_dir = pathlib.Path('I-dont-care'), output_file_prefix = 'osef')) as cacher:
                hash_a = cacher.hash(opts = ['--nvtx'], executable = self.GRAPH, args = ['--bla=42'])
                hash_b = cacher.hash(opts = ['--nvtx'], executable = self.GRAPH, args = ['--bla=42'])

                assert hash_a.digest() == hash_b.digest()

    def test_hash_different(self):
        """
        Test :py:meth:`reprospect.tools.ncu.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir, session = Session(output_dir = pathlib.Path('I-dont-care'), output_file_prefix = 'osef')) as cacher:
                hash_a = cacher.hash(opts = ['--nvtx'], executable = self.GRAPH, args = ['--bla=42'])
                hash_b = cacher.hash(opts = ['--nvtx'], executable = self.SAXPY, args = ['--bla=42'])

                assert hash_a.digest() != hash_b.digest()

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_cache_hit(self):
        """
        The cacher should hit on the second call.
        """
        FILES = ['report-cached.nsys-rep']

        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir, session = Session(output_dir = TMPDIR, output_file_prefix = 'report-cached')) as cacher:
                assert os.listdir(cacher.directory) == ['cache.db']

                results_first = cacher.run(executable = self.GRAPH)

                assert results_first.cached == False

                assert all(x in os.listdir(cacher.session.output_dir) for x in FILES)

                for file in FILES:
                    (cacher.session.output_dir / file).unlink()

                assert sorted(os.listdir(cacher.directory)) == sorted(['cache.db', results_first.digest])
                assert sorted(os.listdir(cacher.directory / results_first.digest)) == sorted(FILES)

                results_second = cacher.run(executable = self.GRAPH)

                assert results_second.cached == True

                assert all(x in os.listdir(cacher.session.output_dir) for x in FILES)

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestReport:
    """
    Test :py:class:`reprospect.tools.nsys.Report`.
    """
    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def report(self) -> Report:
        with Cacher(session = Session(output_dir = TMPDIR, output_file_prefix = 'test-report')) as cacher:
            entry = cacher.run(
                executable = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy',
                cwd = TMPDIR,
                nvtx_capture = '*',
            )

            cacher.export_to_sqlite(entry = entry)

            return Report(db = cacher.session.output_file_sqlite)

    def test_get_events_within_nested_nvtx_ranges(self, report) -> None:
        """
        Check that we can retrieve from a table events that happen in a nested `NVTX` range.
        """
        with report:
            events = report.nvtx_events

            logging.info(events)

            TOP_LEVEL = {
                'application_domain' : 'NvtxDomainCreate',
                'Starting my application.' : 'NvtxMark',
                'outer_useless_range' : 'NvtxStartEndRange',
            }

            roots = events.events[events.events['level'] == 0]

            assert len(roots) == len(TOP_LEVEL)

            assert roots['text'].to_list() == list(TOP_LEVEL.keys())

            for text, event_type_name in TOP_LEVEL.items():
                assert roots[roots['text'] == text].squeeze()['eventTypeName'] == event_type_name

            events = report.get_events(table = 'CUPTI_ACTIVITY_KIND_RUNTIME', accessors = ['outer_useless_range', 'create_streams'])

            assert events['name'].apply(strip_cuda_api_suffix).tolist() == [
                'cuModuleGetLoadingMode',
                'cudaStreamCreate',
                'cudaStreamCreate',
            ]

    class TestNvtxEvents:
        """
        Tests for :py:class:`reprospect.tools.nsys.Report.NvtxEvents`.
        """
        def test_get(self, report) -> None:
            """
            Test :py:meth:`reprospect.tools.nsys.Report.NvtxEvents.get`.
            """
            with report:
                events = report.nvtx_events

                assert len(events.events['text']) == 6

                assert len(events.get(accessors = [])) == 6

                assert len(events.get(accessors = ['outer'])) == 0

                assert events.get(accessors = ['outer_useless_range'])['text'].tolist() == ['outer_useless_range']

                assert events.get(accessors = ['outer_useless_range', 'create_streams'])['text'].tolist() == ['create_streams']

        def test_string_representation(self, report) -> None:
            """
            Test :py:meth:`reprospect.tools.nsys.Report.NvtxEvents.__str__`.
            """
            with report:
                assert str(report.nvtx_events) == """\
NVTX events
├── application_domain (NvtxDomainCreate)
├── Starting my application. (NvtxMark)
└── outer_useless_range (NvtxStartEndRange)
    ├── create_streams (NvtxPushPopRange)
    ├── launch_saxpy_kernel_first_time (NvtxPushPopRange)
    └── launch_saxpy_kernel_second_time (NvtxPushPopRange)
"""

        def test_intricated(self) -> None:
            """
            Use :py:class:`tests.nvtx.test_nvtx.TestNVTX.intricated` to check that we can
            build the hierarchy of `NVTX` events for arbitrarily complicated situations.
            """
            with Cacher(session = Session(output_dir = TMPDIR, output_file_prefix = 'test-report-nvtx')) as cacher:
                entry = cacher.run(
                    executable = pathlib.Path(sys.executable),
                    args = [pathlib.Path(__file__).parent.parent.parent / 'nvtx' / 'test_nvtx.py'],
                    cwd = TMPDIR,
                    nvtx_capture = '*',
                )

                cacher.export_to_sqlite(entry = entry)

                with Report(db = cacher.session.output_file_sqlite) as report:
                    assert len(report.nvtx_events.events) == 7

                    assert report.nvtx_events.get(
                        accessors = ['start-end-level-0', 'push-pop-level-1', 'push-pop-level-2', 'push-pop-level-3']
                    )['text'].tolist() == 3 * ['push-pop-level-3']

                    assert str(report.nvtx_events) == """\
NVTX events
├── intricated (NvtxDomainCreate)
└── start-end-level-0 (NvtxStartEndRange)
    └── push-pop-level-1 (NvtxPushPopRange)
        └── push-pop-level-2 (NvtxPushPopRange)
            ├── push-pop-level-3 (NvtxPushPopRange)
            ├── push-pop-level-3 (NvtxPushPopRange)
            └── push-pop-level-3 (NvtxPushPopRange)
"""
