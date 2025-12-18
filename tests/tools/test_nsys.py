import logging
import os
import pathlib
import sys
import tempfile
import typing
import unittest.mock

import pandas
import pytest
import semantic_version

from reprospect.tools.nsys import (
    Cacher,
    Command,
    Report,
    Session,
    strip_cuda_api_suffix,
)
from reprospect.utils import detect, rich_helpers


class TestTracingResults:
    """
    Test string representation of tracing results.
    """
    CUDA_API_TRACE: typing.Final[pandas.DataFrame] = pandas.DataFrame(
        {
            'Start (us)': (176741.213, 176855.658, 266224.028),
            'Duration (us)': (0.942, 87265.752, 1.753),
            'Name': ('cuModuleGetLoadingMode', 'cudaStreamCreate', 'cudaStreamCreate'),
            'Result': (0, 0, 0),
            'CorrID': (1, 233, 234),
            'Pid': (52655, 52655, 52655),
            'Tid': (52655, 52655, 52655),
            'T-Pri': (20, 20, 20),
            'Thread Name': ('tests_assets_', 'tests_assets_', 'tests_assets_'),
        },
    )

    def test(self) -> None:
        """
        Test string representation of tracing results through conversion to a :py:class:`rich.table.Table`.
        """
        assert rich_helpers.to_string(rich_helpers.df_to_table(self.CUDA_API_TRACE)) == """\
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Start (us) ┃ Duration (us) ┃ Name                   ┃ Result ┃ CorrID ┃ Pid   ┃ Tid   ┃ T-Pri ┃ Thread Name   ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 176741.213 │ 0.942         │ cuModuleGetLoadingMode │ 0      │ 1      │ 52655 │ 52655 │ 20    │ tests_assets_ │
│ 176855.658 │ 87265.752     │ cudaStreamCreate       │ 0      │ 233    │ 52655 │ 52655 │ 20    │ tests_assets_ │
│ 266224.028 │ 1.753         │ cudaStreamCreate       │ 0      │ 234    │ 52655 │ 52655 │ 20    │ tests_assets_ │
└────────────┴───────────────┴────────────────────────┴────────┴────────┴───────┴───────┴───────┴───────────────┘
"""

        single_row = self.CUDA_API_TRACE[self.CUDA_API_TRACE['Name'] == 'cuModuleGetLoadingMode'].squeeze()
        assert rich_helpers.to_string(rich_helpers.ds_to_table(single_row)) == """\
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Start (us) ┃ Duration (us) ┃ Name                   ┃ Result ┃ CorrID ┃ Pid   ┃ Tid   ┃ T-Pri ┃ Thread Name   ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 176741.213 │ 0.942         │ cuModuleGetLoadingMode │ 0      │ 1      │ 52655 │ 52655 │ 20    │ tests_assets_ │
└────────────┴───────────────┴────────────────────────┴────────┴────────┴───────┴───────┴───────┴───────────────┘
"""

class TestCommand:
    """
    Tests for :py:class:`reprospect.tools.nsys.Command`.
    """
    def test_env(self, bindir: pathlib.Path) -> None:
        """
        Check that the environment is handled properly.
        """
        with unittest.mock.patch('subprocess.check_call', return_value = 0) as check_call:
            Command(executable = 'my-executable', output = pathlib.Path('my-output-path.whatever')).run()
            check_call.assert_called_with(args = (
                'nsys', 'profile',
                '--sample=none', '--backtrace=none', '--cpuctxsw=none',
                '--trace=cuda', '--force-overwrite=true', '-o', pathlib.Path('my-output-path.whatever.nsys-rep'),
                'my-executable',
            ), env = None, cwd = None)

        with unittest.mock.patch('subprocess.check_call', return_value = 0) as check_call:
            Command(executable = 'my-executable', output = pathlib.Path('my-output-path.whatever'), env = {'IT_MATTERS': 'ON'}).run(env = {'MY_BASE_ENV': '666'}, cwd = bindir)
            check_call.assert_called_with(args = (
                'nsys', 'profile',
                '--sample=none', '--backtrace=none', '--cpuctxsw=none',
                '--trace=cuda', '--force-overwrite=true', '-o', pathlib.Path('my-output-path.whatever.nsys-rep'),
                'my-executable',
            ), env = {'MY_BASE_ENV': '666', 'IT_MATTERS': 'ON'}, cwd = bindir)

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestSession:
    """
    Test :py:class:`reprospect.tools.nsys.Session`.
    """
    EXECUTABLE: typing.Final[pathlib.Path] = pathlib.Path('tests') / 'assets' / 'tests_assets_saxpy'

    def run(self, bindir: pathlib.Path, cwd: pathlib.Path, nvtx_capture: str | None = None) -> Session:
        ns = Session(
            command = Command(
                executable = bindir / self.EXECUTABLE,
                output = cwd / self.EXECUTABLE.name,
                nvtx_capture = nvtx_capture,
            ),
        )
        ns.run(cwd = cwd)
        ns.export_to_sqlite(cwd = cwd)

        return ns

    def test_cuda_api_trace(self, bindir, workdir) -> None:
        """
        Collect all CUDA API calls of :file:`tests/assets/test_saxpy.cpp`.
        """
        ns = self.run(
            bindir = bindir, cwd = workdir,
            nvtx_capture = "outer_useless_range@application_domain",
        )

        cuda_api_trace = ns.extract_statistical_report(report = 'cuda_api_trace')

        logging.info(f'Report cuda_api_trace:\n{rich_helpers.to_string(rich_helpers.df_to_table(cuda_api_trace))}')

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

    def test_report(self, bindir, workdir) -> None:
        """
        Process the `nsys` report with :py:class:`reprospect.tools.nsys.Report`.
        """
        ns = self.run(bindir = bindir, cwd = workdir)

        cuda_api_trace = ns.extract_statistical_report(report = 'cuda_api_trace')

        with Report(db = ns.command.output.with_suffix('.sqlite')) as report:

            logging.info(f'Tables are {report.tables}.')

            cupti_activity_kind_synchronization = report.table(name = 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')

            logging.info(f'Table CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:\n{rich_helpers.to_string(rich_helpers.df_to_table(cupti_activity_kind_synchronization))}')

            cuda_stream_synchronize = cuda_api_trace[cuda_api_trace['Name'].str.startswith('cudaStreamSynchronize')]

            logging.info(f'Results selected from report cuda_api_trace:\n{rich_helpers.to_string(rich_helpers.df_to_table(cuda_stream_synchronize))}')

            assert len(cuda_stream_synchronize) == 2

            # Each call to 'cudaStreamSynchronize' targets a distinct stream.
            stream_id_a = report.get_correlated_row(src = cuda_stream_synchronize.iloc[0], dst = cupti_activity_kind_synchronization, correlation_src = 'CorrID')['streamId']
            stream_id_b = report.get_correlated_row(src = cuda_stream_synchronize.iloc[1], dst = cupti_activity_kind_synchronization, correlation_src = 'CorrID')['streamId']

            assert stream_id_a != stream_id_b

            # Check that the 'saxpy' kernels ran on stream B.
            cupti_activity_kind_kernel = report.table(name = 'CUPTI_ACTIVITY_KIND_KERNEL')
            saxpy_kernel_first  = cupti_activity_kind_kernel.iloc[0]
            saxpy_kernel_second = cupti_activity_kind_kernel.iloc[1]

            assert saxpy_kernel_first ['streamId'] == stream_id_b
            assert saxpy_kernel_second['streamId'] == stream_id_b

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
            logging.info(f'Table ENUM_CUDA_KERNEL_LAUNCH_TYPE:\n{rich_helpers.to_string(rich_helpers.df_to_table(ENUM_CUDA_KERNEL_LAUNCH_TYPE))}')

            CUDA_KERNEL_LAUNCH_TYPE_REGULAR = report.single_row(data = ENUM_CUDA_KERNEL_LAUNCH_TYPE[ENUM_CUDA_KERNEL_LAUNCH_TYPE['name'] == 'CUDA_KERNEL_LAUNCH_TYPE_REGULAR'])
            logging.info(f'Results selected from table ENUM_CUDA_KERNEL_LAUNCH_TYPE:\n{rich_helpers.to_string(rich_helpers.ds_to_table(CUDA_KERNEL_LAUNCH_TYPE_REGULAR))}')

            assert saxpy_kernel_first ['launchType'] == CUDA_KERNEL_LAUNCH_TYPE_REGULAR['id']
            assert saxpy_kernel_second['launchType'] == CUDA_KERNEL_LAUNCH_TYPE_REGULAR['id']

            # Check 'saxpy' kernels mangled and demangled names.
            stringids = report.table(name = 'StringIds')
            for kernel in [saxpy_kernel_first, saxpy_kernel_second]:
                assert report.single_row(data = stringids[stringids['id'] == kernel['mangledName'  ]])['value'] == '_Z12saxpy_kerneljfPKfPf'
                assert report.single_row(data = stringids[stringids['id'] == kernel['demangledName']])['value'] == 'saxpy_kernel(unsigned int, float, const float *, float *)'

class TestCacher:
    """
    Tests for :py:class:`reprospect.tools.nsys.Cacher`.
    """
    GRAPH: typing.Final[pathlib.Path] = pathlib.Path('tests') / 'assets' / 'tests_assets_graph'
    SAXPY: typing.Final[pathlib.Path] = pathlib.Path('tests') / 'assets' / 'tests_assets_saxpy'

    def test_hash_same(self, bindir) -> None:
        """
        Test :py:meth:`reprospect.tools.nsys.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir) as cacher:
                command = Command(
                    executable = bindir / self.GRAPH,
                    output = pathlib.Path('I-dont-care') / 'osef',
                    opts = ('--nvtx',),
                    args = ('--bla=42',),
                )
                hash_a = cacher.hash(command = command)
                hash_b = cacher.hash(command = command)

                assert hash_a.digest() == hash_b.digest()

    def test_hash_different(self, bindir) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir) as cacher:
                hash_a = cacher.hash(command = Command(executable = bindir / self.GRAPH, output = pathlib.Path('I-dont-care') / 'osef', opts = ('--nvtx',), args = ('--bla=42',)))
                hash_b = cacher.hash(command = Command(executable = bindir / self.SAXPY, output = pathlib.Path('I-dont-care') / 'osef', opts = ('--nvtx',), args = ('--bla=42',)))
                hash_c = cacher.hash(command = Command(executable = bindir / self.SAXPY, output = pathlib.Path('I-dont-care') / 'osef', opts = ('--nvtx',), args = ('--bla=42',), env = {'HELLO': 'WORLD'}))

                assert hash_a.digest() != hash_b.digest()
                assert hash_b.digest() != hash_c.digest()

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_cache_hit(self, bindir, workdir) -> None:
        """
        The cacher should hit on the second call.
        """
        FILES: typing.Final[tuple[str]] = ('report-cached.nsys-rep',)

        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir) as cacher:
                command = Command(
                    executable = bindir / self.GRAPH,
                    output = workdir / FILES[0],
                    opts = ('--env-var=HELLO=WORLD',),
                )

                assert os.listdir(cacher.directory) == ['cache.db']

                results_first = cacher.run(command = command)

                assert results_first.cached is False

                assert all(x in os.listdir(workdir) for x in FILES)

                for file in FILES:
                    (workdir / file).unlink()

                assert sorted(os.listdir(cacher.directory)) == sorted(['cache.db', results_first.digest])
                assert sorted(os.listdir(cacher.directory / results_first.digest)) == sorted(FILES)

                results_second = cacher.run(command = command)

                assert results_second.cached is True

                assert all(x in os.listdir(workdir) for x in FILES)

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestReport:
    """
    Test :py:class:`reprospect.tools.nsys.Report`.
    """
    @pytest.fixture(scope = 'class')
    def report(self, bindir, workdir: pathlib.Path) -> Report:
        with Cacher(directory = workdir) as cacher:
            command = Command(
                executable = bindir / 'tests' / 'assets' / 'tests_assets_saxpy',
                output = workdir / 'test-report.nsys-rep',
                nvtx_capture = '*',
            )

            entry = cacher.run(command = command, cwd = workdir)

            return Report(db = cacher.export_to_sqlite(command = command, entry = entry))

    def test_get_events_within_nested_nvtx_ranges(self, report) -> None:
        """
        Check that we can retrieve from a table events that happen in a nested NVTX range.
        """
        with report:
            events = report.nvtx_events

            logging.info(events)

            TOP_LEVEL = {
                'application_domain': 'NvtxDomainCreate',
                'Starting my application.': 'NvtxMark',
                'outer_useless_range': 'NvtxStartEndRange',
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

            # Stream synchronize events can also be retrieved using NVTX filtering.
            # But their table does not have string IDs to correlate.
            api = report.get_events(table = 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', accessors = ('outer_useless_range', 'initialize_data'), stringids = None)
            assert len(api) == 1

    class TestReportNvtxEvents:
        """
        Tests for :py:class:`reprospect.tools.nsys.ReportNvtxEvents`.
        """
        def test_get(self, report) -> None:
            """
            Test :py:meth:`reprospect.tools.nsys.ReportNvtxEvents.get`.
            """
            with report:
                events = report.nvtx_events

                assert len(events.events['text']) == 7

                assert len(events.get(accessors = ())) == 7

                assert len(events.get(accessors = ('outer',))) == 0

                assert events.get(accessors = ['outer_useless_range'])['text'].tolist() == ['outer_useless_range']

                assert events.get(accessors = ['outer_useless_range', 'create_streams'])['text'].tolist() == ['create_streams']

        def test_string_representation(self, report) -> None:
            """
            Test the string representation of :py:meth:`reprospect.tools.nsys.ReportNvtxEvents`.
            """
            with report:
                assert str(report.nvtx_events) == """\
NVTX events
├── application_domain (NvtxDomainCreate)
├── Starting my application. (NvtxMark)
└── outer_useless_range (NvtxStartEndRange)
    ├── create_streams (NvtxPushPopRange)
    ├── initialize_data (NvtxPushPopRange)
    ├── launch_saxpy_kernel_first_time (NvtxPushPopRange)
    └── launch_saxpy_kernel_second_time (NvtxPushPopRange)
"""

        def test_intricated(self, workdir) -> None:
            """
            Use :py:class:`tests.assets.test_nvtx.TestNVTX.intricated` to check that we can
            build the hierarchy of NVTX events for arbitrarily complicated situations.
            """
            with Cacher(directory = workdir) as cacher:
                command = Command(
                    executable = pathlib.Path(sys.executable),
                    output = workdir / 'test-report-nvtx',
                    nvtx_capture = '*',
                    args = (pathlib.Path(__file__).parent.parent / 'assets' / 'test_nvtx.py',),
                )
                entry = cacher.run(command = command, cwd = workdir)

                with Report(db = cacher.export_to_sqlite(command = command, entry = entry)) as report:
                    assert len(report.nvtx_events.events) == 7

                    assert report.nvtx_events.get(
                        accessors = ['start-end-level-0', 'push-pop-level-1', 'push-pop-level-2', 'push-pop-level-3'],
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
