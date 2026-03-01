"""
Dividing complex numbers is a tricky task.
Indeed, the naïve formula divides by the norm of the denominator,
involving a sum of squares:

.. math::
   :label: naive-complex-div

   \frac{a + bi}{c + di} = \frac{ac + bd}{c^2 + d^2} + \frac{bc - ad}{c^2 + d^2}\,i

To increase the range of values that would not overflow, Kokkos historically
implements a scaling approach, see 
https://github.com/kokkos/kokkos/blob/9174877d49528ab293b6c7f4c6bd932a429b200a/core/src/Kokkos_Complex.hpp#L182-L206.

However, there has been some issues about this:
* https://github.com/kokkos/kokkos/issues/7618
* https://github.com/kokkos/kokkos/issues/7742

Therefore, we started working on it in:
* https://github.com/kokkos/kokkos/issues/8924

The first thing to acknowledge is that the naïve implementation may not be suitable in general,
due to its poor behavior. To quote :cite:`baudin-2012-robust-complex-division-scilab`:

    [...]correct in exact arithmetic, it may fail when we consider floating point numbers.
    Hence, a naive implementation based on the previous formula can easily fail to produce
    an accurate result, [...]

For instance, the following example shows that the :eq:`naive-complex-div` fails at
producing the expected result for:

.. math::
    :label: naive-complex-div-failure

   \frac{1 + i}{1 + i \cdot 2^{1023}} \approx 2^{-1023} - i \cdot 2^{-1023}

Need to scale both numerator and denominator !
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

class Method(StrEnum):
    IEC559  = 'Iec559'
    SCALING = 'Scaling'
    SCALING_BRANCH = 'ScalingBranch'

class TestDivision(CMakeAwareTestCase):
    KOKKOS_TOOLS_NVTX_CONNECTOR_LIB = environment.EnvironmentField(converter=pathlib.Path)
    """Used in :py:meth:`TestNCU.report`."""

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_complex_division'

class TestSASS(TestDivision):
    """
    Binary analysis.
    """
    SIGNATURE: typing.Final[dict[Method, re.Pattern[str]]] = {
        Method.IEC559: re.compile(r'reprospect::examples::kokkos::complex::Iec559<Kokkos::View<Kokkos::complex<double> \[1024\], Kokkos::CudaSpace>>'),
        Method.SCALING: re.compile(r'reprospect::examples::kokkos::complex::Scaling<\(bool\)0, Kokkos::View<Kokkos::complex<double> \[1024\], Kokkos::CudaSpace>>'),
        Method.SCALING_BRANCH: re.compile(r'reprospect::examples::kokkos::complex::Scaling<\(bool\)1, Kokkos::View<Kokkos::complex<double> \[1024\], Kokkos::CudaSpace>>'),
    }

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'examples_kokkos_complex_division.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope='class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file=self.executable, arch=self.arch, sass=True, cwd=self.cwd, cubin=self.cubin.name, demangler=self.demangler)[0]

    @pytest.fixture(scope='class')
    def decoder(self, cuobjdump: CuObjDump) -> dict[Method, Decoder]:
        # logging.info(list(cuobjdump.functions.keys()))
        def get_decoder(method: Method) -> Decoder:
            pattern = self.SIGNATURE[method]
            fctn = cuobjdump.functions[next(sig for sig in cuobjdump.functions if pattern.search(sig) is not None)]
            # logging.info(f'SASS code and resource usage from CuObjDump for {method}:\n{fctn}')
            decoder = Decoder(code=fctn.code)
            # logging.info(f'Decoded SASS code for {method}:\n{decoder}')
            return decoder
        return {method: get_decoder(method) for method in Method}

    def test_instruction_count(self, decoder: dict[Method, Decoder]) -> None:
        """
        Instruction count.

        The :py:data:`Method.IEC559` generates more instructions.
        """
        assert len(decoder[Method.IEC559].instructions) > len(decoder[Method.SCALING_BRANCH].instructions)
        assert len(decoder[Method.SCALING_BRANCH].instructions) > len(decoder[Method.SCALING].instructions)

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestNCU(TestDivision):
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
        Metric(name='gpc__cycles_elapsed.max'),
        Metric(name='sm__inst_executed_pipe_fp64.sum'),
        Metric(name='sm__pipe_fp64_cycles_active.sum'),
        MetricCounter(name='lts__t_sectors_srcunit_tex_op_read_lookup_miss', subs=(MetricCounterRollUp.SUM,)),
    )

    NVTX_INCLUDES: typing.Final[tuple[str, ...]] = ('division',)

    @pytest.fixture(scope='class')
    def report(self) -> Report:
        with Cacher() as cacher:
            command = Command(
                output=self.cwd / 'ncu',
                executable=self.executable,
                metrics=self.METRICS,
                # opts=('--set=full',),
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
    def metrics(self, report: Report) -> dict[Method, ProfilingMetrics]:
        results_in_range = report.extract_results_in_range(
            metrics=self.METRICS,
            includes=self.NVTX_INCLUDES,
        )

        def get_metrics(method: Method) -> ProfilingMetrics:
            kernel, results = results_in_range.query(accessors=(method.name.lower(),)).query_single_next(accessors=())
            _, metrics = results.query_single_next_metrics(accessors=())
            logging.info(f'Kernel {kernel} profiling results for {method} method:\n{results}')
            return metrics

        return {method: get_metrics(method) for method in Method}

    def test_instruction_count(self, metrics: dict[Method, ProfilingMetrics]) -> None:
        logging.info(list(metrics[Method.IEC559].keys()))
        for method in Method:
            logging.info(f"{method} smsp__inst_executed: {metrics[method]['smsp__inst_executed.sum']}")
            logging.info(f"{method} gpc__cycles_elapsed: {metrics[method]['gpc__cycles_elapsed.max']}")
            logging.info(f"{method} sass__inst_executed_per_opcode: {metrics[method]['sass__inst_executed_per_opcode']}")
            logging.info(f"{method} sm__inst_executed_pipe_fp64 {metrics[method]['sm__inst_executed_pipe_fp64.sum']}")
            logging.info(f"{method} sm__pipe_fp64_cycles_active {metrics[method]['sm__pipe_fp64_cycles_active.sum']}")
    
        import pandas as pd
        METRIC_KEYS = [
            "smsp__inst_executed.sum",
            "gpc__cycles_elapsed.max",
            "sass__inst_executed_per_opcode",
            "sm__inst_executed_pipe_fp64.sum",
            "sm__pipe_fp64_cycles_active.sum",
        ]

        COLUMN_LABELS = {
            "smsp__inst_executed.sum":          "Instructions Executed",
            "gpc__cycles_elapsed.max":          "Cycles Elapsed (max)",
            "sass__inst_executed_per_opcode":   "Inst / Opcode",
            "sm__inst_executed_pipe_fp64.sum":  "FP64 Pipe Inst",
            "sm__pipe_fp64_cycles_active.sum":  "FP64 Pipe Cycles Active",
        }

        rows = []
        for method in Method:
            row = {"Method": method.name}
            for key in METRIC_KEYS:
                row[COLUMN_LABELS[key]] = metrics[method].get(key, "N/A")
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Method")

        with open('/workspaces/cuda-helpers/test.html', 'w+') as html:
            html.write(df.to_html(
                classes="metrics-table",
                border=1,
                justify="center",
            ))
    #     """
    #     With specified alignment, half the load/store instructions are executed.
    #     Other instruction counts remain unchanged.
    #     """
    #     # With specified alignment, the overall instruction count is lower.
    #     smsp__inst_executed_sum_default   = metrics[Alignment.DEFAULT  ]['smsp__inst_executed.sum']
    #     smsp__inst_executed_sum_specified = metrics[Alignment.SPECIFIED]['smsp__inst_executed.sum']
    #     assert isinstance(smsp__inst_executed_sum_default, float) and isinstance(smsp__inst_executed_sum_specified, float)
    #     assert smsp__inst_executed_sum_specified < smsp__inst_executed_sum_default

    #     for opcode in metrics[Alignment.SPECIFIED]['sass__inst_executed_per_opcode'].correlated:
    #         if opcode in {'LDG', 'STG'}:
    #             assert metrics[Alignment.DEFAULT  ]['sass__inst_executed_per_opcode'].correlated[opcode] \
    #                 == metrics[Alignment.SPECIFIED]['sass__inst_executed_per_opcode'].correlated[opcode] * 2
    #         elif opcode != 'NOP':
    #             assert metrics[Alignment.DEFAULT  ]['sass__inst_executed_per_opcode'].correlated[opcode] \
    #                 == metrics[Alignment.SPECIFIED]['sass__inst_executed_per_opcode'].correlated[opcode]

    # def test_l1tex_memory_traffic_instruction_count(self, metrics: dict[Alignment, ProfilingMetrics]) -> None:
    #     """
    #     Runtime behavior corresponding to :py:meth:`TestSASS.test_global_memory_instructions`.
    #     """
    #     expt_ldg_count_specified = self.WARP_COUNT * self.LOAD_COUNT
    #     expt_ldg_count_default   = expt_ldg_count_specified * 2

    #     expt_stg_count_specified = self.WARP_COUNT * self.STORE_COUNT
    #     expt_stg_count_default   = expt_stg_count_specified * 2

    #     assert metrics[Alignment.DEFAULT  ]['L1/TEX cache global load instructions sass.sum'] == expt_ldg_count_default
    #     assert metrics[Alignment.SPECIFIED]['L1/TEX cache global load instructions sass.sum'] == expt_ldg_count_specified

    #     assert metrics[Alignment.DEFAULT  ]['L1/TEX cache global store instructions sass.sum'] == expt_stg_count_default
    #     assert metrics[Alignment.SPECIFIED]['L1/TEX cache global store instructions sass.sum'] == expt_stg_count_specified

    #     assert metrics[Alignment.DEFAULT  ]['L1/TEX cache local store instructions sass.sum'] == 0
    #     assert metrics[Alignment.SPECIFIED]['L1/TEX cache local store instructions sass.sum'] == 0

    # def test_l1tex_memory_traffic_sector_count(self, metrics: dict[Alignment, ProfilingMetrics]) -> None:
    #     """
    #     +-----------+-------------------------------------------------------------------------------------------+
    #     | default   | The real parts are read first, and then the imaginary parts,                              |
    #     |           | thus requiring each sector to be read again.                                              |
    #     +-----------+-------------------------------------------------------------------------------------------+
    #     | specified | The loads are coalesced into memory transactions of at least 32 bytes.                    |
    #     |           | Two consecutive complex double values are always loaded together in a single sector load. |
    #     +-----------+-------------------------------------------------------------------------------------------+
    #     """
    #     read_memory = self.LOAD_COUNT * self.ELEMENT_COUNT * self.COMPLEX_DOUBLE_SIZE
    #     sector_count = read_memory / self.SECTOR_SIZE

    #     assert metrics[Alignment.DEFAULT  ]['L1/TEX cache global load sectors.sum'] == sector_count * 2
    #     assert metrics[Alignment.SPECIFIED]['L1/TEX cache global load sectors.sum'] == sector_count

    # def test_l2_memory_traffic_sector_count(self, metrics: dict[Alignment, ProfilingMetrics]) -> None:
    #     """
    #     The traffic out to L2 and out to DRAM is the same with both default and specialized alignments.

    #     With the default alignment, there are two consecutive loads, but the second load concerns
    #     the same sector as the first load and can thus be expected to hit in L1 cache.
    #     """
    #     keys_sectors_lookup_misses = (
    #         'L1/TEX cache global load sectors miss.sum',
    #         'lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum',
    #     )

    #     expt_sectors_lookup_misses = self.LOAD_COUNT * self.ELEMENT_COUNT * self.COMPLEX_DOUBLE_SIZE / self.SECTOR_SIZE

    #     for key in keys_sectors_lookup_misses:
    #         for alignment in (Alignment.DEFAULT, Alignment.SPECIFIED):
    #             assert metrics[alignment][key] == expt_sectors_lookup_misses
