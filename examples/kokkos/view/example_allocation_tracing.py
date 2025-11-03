import logging
import math
import pathlib
import sys
import typing

import numpy
import pytest
import typeguard

import reprospect

from reprospect.test  import environment
from reprospect.tools import nsys
from reprospect.utils import detect

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Memory(StrEnum):
    DEVICE = 'DEVICE'
    SHARED = 'MANAGED'

class TestAllocation(reprospect.CMakeAwareTestCase):
    """
    Explore the behavior of `Kokkos::View` allocation under different scenarios.
    """
    KOKKOS_TOOLS_NVTX_CONNECTOR_LIB = environment.EnvironmentField(converter = pathlib.Path)

    @classmethod
    @override
    @typeguard.typechecked
    def get_target_name(cls) -> str:
        return 'examples_kokkos_view_allocation_tracing'

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
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
                nvtx_capture = 'AllocationProfiling',
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

    @staticmethod
    @typeguard.typechecked
    def get_memory_id(report : nsys.Report, memory : Memory) -> numpy.int64:
        """
        Retrieve the `id` from `ENUM_CUDA_MEM_KIND` whose `name` matches `memory`.
        """
        enum_cuda_mem_kind = report.table(name = 'ENUM_CUDA_MEM_KIND', dtype = {'name' : str, 'id' : int})

        return report.single_row(
            data = enum_cuda_mem_kind[enum_cuda_mem_kind['name'] == f'CUDA_MEMOPR_MEMORY_KIND_{memory}']
        )['id']

    def checks(self, *,
        report : nsys.Report,
        expt_cuda_api_calls_allocation : typing.Iterable[str],
        expt_cuda_api_calls_deallocation : typing.Iterable[str],
        selectors : dict[str, nsys.Report.PatternSelector],
        memory : Memory,
        size : int,
    ) -> None:
        match memory:
            case Memory.SHARED:
                memory_space = 'Kokkos::CudaUVMSpace'
            case Memory.DEVICE:
                memory_space = 'Kokkos::CudaSpace'
            case _:
                raise ValueError(f'unsupported memory {memory}')

        with report:
            logging.info(report.nvtx_events)

            cuda_api_trace_allocation   = report.get_events(table = 'CUPTI_ACTIVITY_KIND_RUNTIME', accessors = ['AllocationProfiling', memory_space, str(size), 'allocation'])
            cuda_api_trace_deallocation = report.get_events(table = 'CUPTI_ACTIVITY_KIND_RUNTIME', accessors = ['AllocationProfiling', memory_space, str(size), 'deallocation'])

            logging.info(f"Events during allocation: {cuda_api_trace_allocation['name']}.")
            logging.info(f"Events during deallocation: {cuda_api_trace_deallocation['name']}.")

            # Allocation goes through the expected Cuda API calls.
            assert len(cuda_api_trace_allocation['name']) == len(expt_cuda_api_calls_allocation)
            assert cuda_api_trace_allocation['name'].apply(nsys.strip_cuda_api_suffix).tolist() == expt_cuda_api_calls_allocation

            # Deallocation goes through the expected Cuda API calls.
            assert len(cuda_api_trace_deallocation['name']) == len(expt_cuda_api_calls_deallocation)
            assert cuda_api_trace_deallocation['name'].apply(nsys.strip_cuda_api_suffix).tolist() == expt_cuda_api_calls_deallocation

            # Check size and kind of allocation.
            cuda_gpu_memory_usage_events = report.table(name = 'CUDA_GPU_MEMORY_USAGE_EVENTS')

            malloc = report.get_correlated_row(src = cuda_api_trace_allocation, selector = selectors['malloc'], dst = cuda_gpu_memory_usage_events)

            assert malloc['bytes']   == size + self.HEADER_SIZE
            assert malloc['memKind'] == self.get_memory_id(report = report, memory = memory)

            match memory:
                case Memory.DEVICE:
                    # Kokkos copies the shared allocation header.
                    cupti_activity_kind_memcpy = report.table(name = 'CUPTI_ACTIVITY_KIND_MEMCPY')

                    memcpy = report.get_correlated_row(src = cuda_api_trace_allocation, selector = selectors['memcpy'], dst = cupti_activity_kind_memcpy)

                    assert memcpy['bytes'] == self.HEADER_SIZE
                    if not math.isnan(malloc['streamId']):
                        assert 'cudaMallocAsync' in expt_cuda_api_calls_allocation
                        assert memcpy['streamId'] == malloc['streamId']
                    else:
                        assert 'cudaMalloc' in expt_cuda_api_calls_allocation

                case Memory.SHARED:
                    # There is no need to copy the header when the memory is managed.
                    assert selectors['memcpy'] is None

                case _:
                    raise ValueError(f'unsupported memory {memory}')

            # Check that 'malloc' and 'free' work on the same pointer address.
            free = report.get_correlated_row(src = cuda_api_trace_deallocation, selector = selectors['free'], dst = cuda_gpu_memory_usage_events)

            assert malloc['address']  == free['address']
            assert malloc['streamId'] != free['streamId']

    def test_under_39000_CudaSpace(self, report : nsys.Report) -> None:
        """
        Check what happens under the threshold for `Kokkos::CudaSpace` (requested size is 39000).
        """
        self.checks(
            report = report,
            size = 39000,
            memory = Memory.DEVICE,
            expt_cuda_api_calls_allocation = [
                'cudaMalloc',
                'cudaMemcpyAsync',
            ],
            expt_cuda_api_calls_deallocation = [
                'cudaFree',
            ],
            selectors = {
                'malloc' : report.PatternSelector(column = 'name', pattern = r'^cudaMalloc'),
                'memcpy' : report.PatternSelector(column = 'name', pattern = r'^cudaMemcpyAsync'),
                'free'   : report.PatternSelector(column = 'name', pattern = r'^cudaFree'),
            },
        )

    def test_under_39000_CudaUVMSpace(self, report : nsys.Report) -> None:
        """
        Check what happens under the threshold for `Kokkos::CudaUVMSpace` (requested size is 39000).
        """
        self.checks(
            report = report,
            size = 39000,
            memory = Memory.SHARED,
            expt_cuda_api_calls_allocation = [
                'cudaDeviceSynchronize',
                'cudaMallocManaged',
                'cudaDeviceSynchronize',
            ],
            expt_cuda_api_calls_deallocation = [
                'cudaDeviceSynchronize',
                'cudaFree',
                'cudaDeviceSynchronize',
            ],
            selectors = {
                'malloc' : report.PatternSelector(column = 'name', pattern = r'^cudaMallocManaged'),
                'memcpy' : None,
                'free'   : report.PatternSelector(column = 'name', pattern = r'^cudaFree'),
            },
        )

    def test_above_41000_CudaSpace(self, report : nsys.Report) -> None:
        """
        Check what happens above the threshold for `Kokkos::CudaSpace` (requested size is 41000).
        """
        self.checks(
            report = report,
            size = 41000,
            memory = Memory.DEVICE,
            expt_cuda_api_calls_allocation = [
                'cudaMallocAsync',
                'cudaStreamSynchronize',
                'cudaMemcpyAsync',
            ],
            expt_cuda_api_calls_deallocation = [
                'cudaDeviceSynchronize',
                'cudaFreeAsync',
                'cudaDeviceSynchronize',
            ],
            selectors = {
                'malloc' : report.PatternSelector(column = 'name', pattern = r'^cudaMallocAsync'),
                'memcpy' : report.PatternSelector(column = 'name', pattern = r'^cudaMemcpyAsync'),
                'free'   : report.PatternSelector(column = 'name', pattern = r'^cudaFreeAsync'),
            },
        )

    def test_above_41000_CudaUVMSpace(self, report : nsys.Report) -> None:
        """
        Check what happens above the threshold for `Kokkos::CudaUVMSpace` (requested size is 41000).
        """
        self.checks(
            report = report,
            size = 41000,
            memory = Memory.SHARED,
            expt_cuda_api_calls_allocation = [
                'cudaDeviceSynchronize',
                'cudaMallocManaged',
                'cudaDeviceSynchronize',
            ],
            expt_cuda_api_calls_deallocation = [
                'cudaDeviceSynchronize',
                'cudaFree',
                'cudaDeviceSynchronize',
            ],
            selectors = {
                'malloc' : report.PatternSelector(column = 'name', pattern = r'^cudaMallocManaged'),
                'memcpy' : None,
                'free'   : report.PatternSelector(column = 'name', pattern = r'^cudaFree'),
            },
        )
