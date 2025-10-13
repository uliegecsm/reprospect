import pathlib

import pandas
import pytest
import typeguard

import reprospect

from reprospect.tools import nsys

class TestAllocation(reprospect.TestCase):
    """
    Explore the behavior of `Kokkos::View` allocation under different scenarios.
    """
    NAME = 'examples_kokkos_view_allocation'

    @property
    @typeguard.typechecked
    def executable(self) -> pathlib.Path:
        return self.CMAKE_BINARY_DIR / 'examples' / 'kokkos' / 'view' / self.NAME

class TestNSYS(TestAllocation):
    """
    `nsys`-focused analysis.
    """
    HEADER_SIZE = 128
    """Size of the `Kokkos::Impl::SharedAllocationHeader` type, see https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/impl/Kokkos_SharedAlloc.hpp#L23."""

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def session(self) -> nsys.Session:
        """
        Analyse with `nsys`, use :py:class:`reprospect.tools.nsys.Cacher`.
        """
        with nsys.Cacher(
            session = nsys.Session(
                output_dir = self.cwd,
                output_file_prefix = self.executable.name,
            )
        ) as cacher:
            entry = cacher.run(
                nvtx_capture = 'Allocation',
                opts = ['--cuda-memory-usage=true'],
                executable = self.executable,
                args = [
                    f"--kokkos-tools-libs={self.KOKKOS_TOOLS_NVTX_CONNECTOR_LIB}",
                ],
                cwd = self.cwd,
            )

            cacher.export_to_sqlite(entry)

            return cacher.session

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def report(self, session : nsys.Session) -> nsys.Report:
        return nsys.Report(db = session.output_file_sqlite)

    @pytest.fixture(scope = 'class')
    @typeguard.typechecked
    def memory_kind(self, report : nsys.Report) -> pandas.Series:
        """
        Retrieve the expected memory kind for `CUDA_MEMOPR_MEMORY_KIND_DEVICE`.
        """
        with report:
            enum_cuda_mem_kind = report.table(name = 'ENUM_CUDA_MEM_KIND')

        return report.single_row(data = enum_cuda_mem_kind[enum_cuda_mem_kind['name'] == 'CUDA_MEMOPR_MEMORY_KIND_DEVICE'])

    def test_under_39000(self, session : nsys.Session, report : nsys.Report, memory_kind) -> None:
        """
        Check what happens under the threshold (requested size is 39000).
        """
        cuda_api_trace_allocation   = session.extract_statistical_report(report = 'cuda_api_trace', filter_nvtx = f'allocation/0')
        cuda_api_trace_deallocation = session.extract_statistical_report(report = 'cuda_api_trace', filter_nvtx = f'deallocation/0')

        # Allocation goes through these calls.
        exptd_api_calls_allocation = [
            'cudaMalloc',
            'cudaMemcpyAsync',
        ]
        assert len(cuda_api_trace_allocation['Name']) == len(exptd_api_calls_allocation)
        assert cuda_api_trace_allocation['Name'].apply(nsys.strip_cuda_api_suffix).tolist() == exptd_api_calls_allocation

        # Deallocation goes through these calls.
        exptd_api_calls_deallocation = ['cudaFree']
        assert len(cuda_api_trace_deallocation['Name']) == len(exptd_api_calls_deallocation)
        assert cuda_api_trace_deallocation['Name'].apply(nsys.strip_cuda_api_suffix).tolist() == exptd_api_calls_deallocation

        with report:
            cuda_gpu_memory_usage_events = report.table(name = 'CUDA_GPU_MEMORY_USAGE_EVENTS')
            cupti_activity_kind_memcpy   = report.table(name = 'CUPTI_ACTIVITY_KIND_MEMCPY')

            # Check size of allocation.
            cuda_malloc = report.get_correlated_row(src = cuda_api_trace_allocation, selector = report.PatternSelector(pattern = r'^cudaMalloc'), dst = cuda_gpu_memory_usage_events)

            assert cuda_malloc['bytes'] == 39000 + self.HEADER_SIZE
            assert cuda_malloc['memKind'] == memory_kind['id']

            # There is a copy of the shared allocation header.
            cuda_memcpy_async = report.get_correlated_row(src = cuda_api_trace_allocation, selector = report.PatternSelector(pattern = r'^cudaMemcpyAsync'), dst = cupti_activity_kind_memcpy)

            assert cuda_memcpy_async['bytes'] == self.HEADER_SIZE

            # Check that 'malloc' and 'free' work on the same pointer address.
            cuda_free = report.get_correlated_row(src = cuda_api_trace_deallocation, selector = report.PatternSelector(pattern = r'^cudaFree'), dst = cuda_gpu_memory_usage_events)

            assert cuda_malloc['address'] == cuda_free['address']

    def test_above_41000(self, session : nsys.Session, report : nsys.Report, memory_kind) -> None:
        """
        Check what happens above the threshold (requested size is 41000).
        """
        cuda_api_trace_allocation   = session.extract_statistical_report(report = 'cuda_api_trace', filter_nvtx = f'allocation/1')
        cuda_api_trace_deallocation = session.extract_statistical_report(report = 'cuda_api_trace', filter_nvtx = f'deallocation/1')

        # Allocation goes through these calls.
        exptd_api_calls_allocation = [
            'cudaMallocAsync',
            'cudaStreamSynchronize',
            'cudaMemcpyAsync',
        ]
        assert len(cuda_api_trace_allocation['Name']) == len(exptd_api_calls_allocation)
        assert cuda_api_trace_allocation['Name'].apply(nsys.strip_cuda_api_suffix).tolist() == exptd_api_calls_allocation

        # Deallocation goes through these calls.
        exptd_api_calls_deallocation = [
            'cudaDeviceSynchronize',
            'cudaFreeAsync',
            'cudaDeviceSynchronize',
        ]
        assert len(cuda_api_trace_deallocation['Name']) == len(exptd_api_calls_deallocation)
        assert cuda_api_trace_deallocation['Name'].apply(nsys.strip_cuda_api_suffix).tolist() == exptd_api_calls_deallocation

        with report:
            cuda_gpu_memory_usage_events = report.table(name = 'CUDA_GPU_MEMORY_USAGE_EVENTS')
            cupti_activity_kind_memcpy   = report.table(name = 'CUPTI_ACTIVITY_KIND_MEMCPY')

            # Check size of allocation.
            cuda_malloc_async = report.get_correlated_row(src = cuda_api_trace_allocation, selector = report.PatternSelector(pattern = r'^cudaMallocAsync'), dst = cuda_gpu_memory_usage_events)

            assert cuda_malloc_async['bytes'] == 41000 + self.HEADER_SIZE
            assert cuda_malloc_async['memKind'] == memory_kind['id']

            # There is a copy of the shared allocation header.
            cuda_memcpy_async = report.get_correlated_row(src = cuda_api_trace_allocation, selector = report.PatternSelector(pattern = r'^cudaMemcpyAsync'), dst = cupti_activity_kind_memcpy)

            assert cuda_memcpy_async['bytes'] == self.HEADER_SIZE
            assert cuda_memcpy_async['streamId'] == cuda_malloc_async['streamId']

            # Check that 'malloc' and 'free' work on the same pointer address.
            cuda_free_async = report.get_correlated_row(src = cuda_api_trace_deallocation, selector = report.PatternSelector(pattern = r'^cudaFreeAsync'), dst = cuda_gpu_memory_usage_events)

            assert cuda_malloc_async['address']  == cuda_free_async['address']
            assert cuda_malloc_async['streamId'] != cuda_free_async['streamId']
