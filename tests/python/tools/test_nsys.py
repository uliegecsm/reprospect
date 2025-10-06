import logging
import os
import pathlib
import tempfile
import typing

import pandas
import semantic_version
import typeguard

from reprospect.tools.nsys import Report, Session, strip_cuda_api_suffix, Cacher

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

class TestSession:
    """
    Test :py:class:`reprospect.tools.nsys.Session`.
    """
    EXECUTABLE = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy' if 'CMAKE_BINARY_DIR' in os.environ else None

    @staticmethod
    @typeguard.typechecked
    def single_row_to_dict(*, data : pandas.DataFrame) -> dict[str, typing.Any]:
        """
        Check that `data` has one row, and convert it to a `dict`.
        """
        assert len(data) == 1, data
        return data.squeeze().to_dict()

    def run(self) -> Session:
        assert self.EXECUTABLE.is_file()

        ns = Session(
            output_dir = TMPDIR,
            output_file_prefix = self.EXECUTABLE.name,
        )

        ns.run(
            executable = self.EXECUTABLE,
            cwd = TMPDIR,
        )

        ns.export_to_sqlite()

        return ns

    def test_cuda_api_trace(self):
        """
        Collect all `Cuda` API calls of :file:`tests/cpp/cuda/test_saxpy.cpp`.
        """
        ns = self.run()

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
            stream_id_A = self.single_row_to_dict(data = cupti_activity_kind_synchronization[cupti_activity_kind_synchronization['correlationId'] == cuda_stream_synchronize.iloc[0]['CorrID']])['streamId']
            stream_id_B = self.single_row_to_dict(data = cupti_activity_kind_synchronization[cupti_activity_kind_synchronization['correlationId'] == cuda_stream_synchronize.iloc[1]['CorrID']])['streamId']

            assert stream_id_A != stream_id_B

            # Check that the 'saxpy' kernel ran on stream B.
            cupti_activity_kind_kernel = report.table(name = 'CUPTI_ACTIVITY_KIND_KERNEL')
            saxpy_kernel = self.single_row_to_dict(data = cupti_activity_kind_kernel)

            assert saxpy_kernel['streamId'] == stream_id_B

            # Check 'saxpy' kernel launch block/grid configuration.
            assert saxpy_kernel['gridX'] == 8
            assert saxpy_kernel['gridY'] == 1
            assert saxpy_kernel['gridZ'] == 1

            assert saxpy_kernel['blockX'] == 128
            assert saxpy_kernel['blockY'] == 1
            assert saxpy_kernel['blockZ'] == 1

            # Check 'saxpy' kernel launch type.
            ENUM_CUDA_KERNEL_LAUNCH_TYPE = report.table(name = 'ENUM_CUDA_KERNEL_LAUNCH_TYPE')

            assert saxpy_kernel['launchType'] == self.single_row_to_dict(data = ENUM_CUDA_KERNEL_LAUNCH_TYPE[ENUM_CUDA_KERNEL_LAUNCH_TYPE['name'] == 'CUDA_KERNEL_LAUNCH_TYPE_REGULAR'])['id']

            # Check 'saxpy' kernel mangled and demangled names.
            stringids = report.table(name = 'StringIds')
            assert self.single_row_to_dict(data = stringids[stringids['id'] == saxpy_kernel['mangledName'  ]])['value'] == '_Z12saxpy_kerneljfPKfPf'
            assert self.single_row_to_dict(data = stringids[stringids['id'] == saxpy_kernel['demangledName']])['value'] == 'saxpy_kernel(unsigned int, float, const float *, float *)'

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
