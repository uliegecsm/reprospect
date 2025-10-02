import logging
import os
import pathlib
import typing

import pandas
import semantic_version
import typeguard

from reprospect.tools.nsys import Report, Session, strip_cuda_api_suffix

class TestSession:
    """
    Test :py:class:`reprospect.tools.nsys.Session`.
    """
    TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None
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
            output_dir = self.TMPDIR,
            output_file_prefix = self.EXECUTABLE.name,
        )

        ns.run(
            cmd = [self.EXECUTABLE],
            cwd = self.TMPDIR,
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
