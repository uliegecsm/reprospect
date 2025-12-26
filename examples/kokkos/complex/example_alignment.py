"""
Because :code:`std::complex` lacks :code:`__host__ __device__` annotations, the :code:`Kokkos` library
provides its own annotated implementation :code:`Kokkos::complex`.

However, :code:`Kokkos::complex` is *not* a drop-in replacement for :code:`std::complex`.
Beyond the annotations, there are other differences, including:

* **Alignment:** Whereas :code:`std::complex` is aligned to only the size of the underlying real type,
  :code:`Kokkos::complex` is aligned to twice the size of the underlying real type by default
  (`PR #2259 <https://github.com/kokkos/kokkos/pull/2259>`_).
* **Division operator:** :code:`Kokkos::complex` carries out the complex number division in a special way
  that provides better overflow protection (`Issue #7618 <https://github.com/kokkos/kokkos/issues/7618>`_).
* **Operator overloads:** There are also differences in allowed member and non-member operator overloads
  (`PR #8212 <https://github.com/kokkos/kokkos/pull/8212>`_).

This example analyzes the impact of the specified alignment of :code:`Kokkos::complex`.
It analyzes a kernel that reads from and writes to contiguous arrays of 128-bit structures:

    +---------------------------------+---------------------------------+------------------------+-------------------------+
    |                                 | Type                            | :code:`sizeof` [bytes] | :code:`alignof` [bytes] |
    +=================================+=================================+========================+=========================+
    | :py:const:`Alignment.DEFAULT`   | custom                          | 16                     | 8                       |
    +---------------------------------+---------------------------------+------------------------+-------------------------+
    | :py:const:`Alignment.SPECIFIED` | :code:`Kokkos::complex<double>` | 16                     | 16                      |
    +---------------------------------+---------------------------------+------------------------+-------------------------+

The example proceeds in two steps:

#. :py:class:`TestSASS` examines compiler-generated instructions from the binary.

   It shows that with specified alignment, reading or writing a :code:`Kokkos::complex<double>`
   value in global memory compiles to 128-bit instructions.

   By contrast, with the default alignment, the compiler generates two successive 64-bit instructions, the first
   for the real part, and the second for the imaginary part.

#. :py:class:`TestNCU` performs a kernel profiling.

   * Default alignment leads to twice more sector requests than specified alignment
     (see :py:meth:`TestNCU.test_l1tex_memory_traffic_sector_count`).

   * However, the additional sector requests with default alignment hit in L1 cache
     (see :py:meth:`TestNCU.test_l2_memory_traffic_sector_count`).

See also:

* https://github.com/NVIDIA/cuda-samples/blob/c94ff366aed18c797b8a85dfaac7817b0228b420/Samples/6_Performance/alignedTypes/alignedTypes.cu
"""

import logging
import pathlib
import re
import sys
import typing

import pytest

from reprospect.test import environment
from reprospect.test.case import CMakeAwareTestCase
from reprospect.test.sass.instruction import LoadGlobalMatcher, StoreGlobalMatcher
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.ncu import (
    Cacher,
    Command,
    L1TEXCache,
    Metric,
    MetricCorrelation,
    MetricCounter,
    MetricCounterRollUp,
    ProfilingMetrics,
    Report,
)
from reprospect.tools.sass import Decoder
from reprospect.utils import detect

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Alignment(StrEnum):
    DEFAULT   = 'default'
    SPECIFIED = 'specified'

class TestAlignment(CMakeAwareTestCase):
    """
    The kernel under study loads 3 arrays of :py:attr:`ELEMENT_COUNT` elements
    and writes to 1 array of :py:attr:`ELEMENT_COUNT` elements.

    The threads work on consecutive elements of the arrays, in such a way that each
    thread reads a total of 3 elements and writes 1 element.
    """
    LOAD_COUNT: typing.Final[int] = 3
    STORE_COUNT: typing.Final[int] = 1
    ELEMENT_COUNT: typing.Final[int] = 1024
    WARP_SIZE: typing.Final[int] = 32
    WARP_COUNT: typing.Final[int] = ELEMENT_COUNT // WARP_SIZE
    COMPLEX_DOUBLE_SIZE: typing.Final[int] = 16
    SECTOR_SIZE: typing.Final[int] = 32

    KOKKOS_TOOLS_NVTX_CONNECTOR_LIB = environment.EnvironmentField(converter=pathlib.Path)
    """Used in :py:meth:`TestNCU.report`."""

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_complex_alignment'

class TestSASS(TestAlignment):
    """
    Binary analysis.
    """
    SIGNATURE: typing.Final[dict[Alignment, re.Pattern[str]]] = {
        Alignment.DEFAULT:   re.compile(r'MulAddFunctor<Kokkos::View<reprospect::examples::kokkos::complex::Complex<double>\s*\*, Kokkos::CudaSpace>>'),
        Alignment.SPECIFIED: re.compile(r'MulAddFunctor<Kokkos::View<Kokkos::complex<double>\s*\*, Kokkos::CudaSpace>>'),
    }

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'examples_kokkos_complex_alignment.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope='class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file=self.executable, arch=self.arch, sass=True, cwd=self.cwd, cubin=self.cubin.name, demangler=self.demangler)[0]

    @pytest.fixture(scope='class')
    def decoder(self, cuobjdump: CuObjDump) -> dict[Alignment, Decoder]:
        def get_decoder(alignment: Alignment) -> Decoder:
            pattern = self.SIGNATURE[alignment]
            fctn = cuobjdump.functions[next(sig for sig in cuobjdump.functions if pattern.search(sig) is not None)]
            logging.info(f'SASS code and resource usage from CuObjDump for {alignment} alignment:\n{fctn}')
            decoder = Decoder(code=fctn.code)
            logging.info(f'Decoded SASS code for {alignment} alignment:\n{decoder}')
            return decoder
        return {alignment: get_decoder(alignment) for alignment in Alignment}

    def test_global_memory_instructions(self, decoder: dict[Alignment, Decoder]) -> None:
        """
        Check the type and count of global load and store instructions used.

        +-----------+----------------------------------------------------------------------+
        | default   | Each element is loaded/stored with 2 instructions of 64 bits.        |
        +-----------+----------------------------------------------------------------------+
        | specified | Each element is loaded/stored with a single instruction of 128 bits. |
        +-----------+----------------------------------------------------------------------+
        """
        expt_ldg_count_specified = self.LOAD_COUNT
        expt_ldg_count_default   = expt_ldg_count_specified * 2

        assert sum(1 for inst in decoder[Alignment.DEFAULT]  .instructions if LoadGlobalMatcher(arch=self.arch, size=64 )(inst)) == expt_ldg_count_default
        assert sum(1 for inst in decoder[Alignment.SPECIFIED].instructions if LoadGlobalMatcher(arch=self.arch, size=128)(inst)) == expt_ldg_count_specified

        expt_stg_count_specified = self.STORE_COUNT
        expt_stg_count_default   = expt_stg_count_specified * 2

        assert sum(1 for inst in decoder[Alignment.DEFAULT]  .instructions if StoreGlobalMatcher(arch=self.arch, size=64 )(inst)) == expt_stg_count_default
        assert sum(1 for inst in decoder[Alignment.SPECIFIED].instructions if StoreGlobalMatcher(arch=self.arch, size=128)(inst)) == expt_stg_count_specified

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestNCU(TestAlignment):
    """
    Kernel profiling.
    """
    METRICS: tuple[Metric | MetricCorrelation, ...] = (
        # Overall instruction count.
        MetricCounter(name='smsp__inst_executed', subs=(MetricCounterRollUp.SUM,)),
        # Specific instruction counts (LDG and STG and others).
        MetricCorrelation(name='sass__inst_executed_per_opcode'),
        # Memory traffic.
        L1TEXCache.GlobalLoad.Instructions.create(),
        L1TEXCache.GlobalLoad.Sectors.create(),
        L1TEXCache.GlobalLoad.SectorMisses.create(),
        L1TEXCache.GlobalStore.Instructions.create(),
        L1TEXCache.LocalStore.Instructions.create(),
        MetricCounter(name='lts__t_sectors_srcunit_tex_op_read_lookup_miss', subs=(MetricCounterRollUp.SUM,)),
    )

    NVTX_INCLUDES: typing.Final[tuple[str, ...]] = ('alignment',)

    @pytest.fixture(scope='class')
    def report(self) -> Report:
        with Cacher() as cacher:
            command = Command(
                output=self.cwd / 'ncu',
                executable=self.executable,
                metrics=self.METRICS,
                nvtx_includes=self.NVTX_INCLUDES,
                args=(
                    f'--kokkos-tools-libs={self.KOKKOS_TOOLS_NVTX_CONNECTOR_LIB}',
                ),
            )
            cacher.run(
                command=command,
                cwd=self.cwd,
                retries=5,
            )
        return Report(command=command)

    @pytest.fixture(scope='class')
    def metrics(self, report: Report) -> dict[Alignment, ProfilingMetrics]:
        results_in_range = report.extract_results_in_range(
            metrics=self.METRICS,
            includes=self.NVTX_INCLUDES,
        )

        def get_metrics(alignment: Alignment) -> ProfilingMetrics:
            results = results_in_range.query(accessors=(alignment.value, 'multiply and add view elements'))
            _, metrics = results.query_single_next_metrics(accessors=())
            logging.info(f'Kernel profiling results for {alignment} alignment:\n{results}')
            return metrics

        return {alignment: get_metrics(alignment) for alignment in Alignment}

    def test_instruction_count(self, metrics: dict[Alignment, ProfilingMetrics]) -> None:
        """
        With specified alignment, half the load/store instructions are executed.
        Other instruction counts remain unchanged.
        """
        # With specified alignment, the overall instruction count is lower.
        smsp__inst_executed_sum_default   = metrics[Alignment.DEFAULT  ]['smsp__inst_executed.sum']
        smsp__inst_executed_sum_specified = metrics[Alignment.SPECIFIED]['smsp__inst_executed.sum']
        assert isinstance(smsp__inst_executed_sum_default, float) and isinstance(smsp__inst_executed_sum_specified, float)
        assert smsp__inst_executed_sum_specified < smsp__inst_executed_sum_default

        for opcode in metrics[Alignment.SPECIFIED]['sass__inst_executed_per_opcode'].correlated:
            if opcode in {'LDG', 'STG'}:
                assert metrics[Alignment.DEFAULT  ]['sass__inst_executed_per_opcode'].correlated[opcode] \
                    == metrics[Alignment.SPECIFIED]['sass__inst_executed_per_opcode'].correlated[opcode] * 2
            elif opcode != 'NOP':
                assert metrics[Alignment.DEFAULT  ]['sass__inst_executed_per_opcode'].correlated[opcode] \
                    == metrics[Alignment.SPECIFIED]['sass__inst_executed_per_opcode'].correlated[opcode]

    def test_l1tex_memory_traffic_instruction_count(self, metrics: dict[Alignment, ProfilingMetrics]) -> None:
        """
        Runtime behavior corresponding to :py:meth:`TestSASS.test_global_memory_instructions`.
        """
        expt_ldg_count_specified = self.WARP_COUNT * self.LOAD_COUNT
        expt_ldg_count_default   = expt_ldg_count_specified * 2

        expt_stg_count_specified = self.WARP_COUNT * self.STORE_COUNT
        expt_stg_count_default   = expt_stg_count_specified * 2

        assert metrics[Alignment.DEFAULT  ]['L1/TEX cache global load instructions sass.sum'] == expt_ldg_count_default
        assert metrics[Alignment.SPECIFIED]['L1/TEX cache global load instructions sass.sum'] == expt_ldg_count_specified

        assert metrics[Alignment.DEFAULT  ]['L1/TEX cache global store instructions sass.sum'] == expt_stg_count_default
        assert metrics[Alignment.SPECIFIED]['L1/TEX cache global store instructions sass.sum'] == expt_stg_count_specified

        assert metrics[Alignment.DEFAULT  ]['L1/TEX cache local store instructions sass.sum'] == 0
        assert metrics[Alignment.SPECIFIED]['L1/TEX cache local store instructions sass.sum'] == 0

    def test_l1tex_memory_traffic_sector_count(self, metrics: dict[Alignment, ProfilingMetrics]) -> None:
        """
        +-----------+-------------------------------------------------------------------------------------------+
        | default   | The real parts are read first, and then the imaginary parts,                              |
        |           | thus requiring each sector to be read again.                                              |
        +-----------+-------------------------------------------------------------------------------------------+
        | specified | The loads are coalesced into memory transactions of at least 32 bytes.                    |
        |           | Two consecutive complex double values are always loaded together in a single sector load. |
        +-----------+-------------------------------------------------------------------------------------------+
        """
        read_memory = self.LOAD_COUNT * self.ELEMENT_COUNT * self.COMPLEX_DOUBLE_SIZE
        sector_count = read_memory / self.SECTOR_SIZE

        assert metrics[Alignment.DEFAULT  ]['L1/TEX cache global load sectors.sum'] == sector_count * 2
        assert metrics[Alignment.SPECIFIED]['L1/TEX cache global load sectors.sum'] == sector_count

    def test_l2_memory_traffic_sector_count(self, metrics: dict[Alignment, ProfilingMetrics]) -> None:
        """
        The traffic out to L2 and out to DRAM is the same with both default and specialized alignments.

        With the default alignment, there are two consecutive loads, but the second load concerns
        the same sector as the first load and can thus be expected to hit in L1 cache.
        """
        keys_sectors_lookup_misses = (
            'L1/TEX cache global load sectors miss.sum',
            'lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum',
        )

        expt_sectors_lookup_misses = self.LOAD_COUNT * self.ELEMENT_COUNT * self.COMPLEX_DOUBLE_SIZE / self.SECTOR_SIZE

        for key in keys_sectors_lookup_misses:
            for alignment in (Alignment.DEFAULT, Alignment.SPECIFIED):
                assert metrics[alignment][key] == expt_sectors_lookup_misses
