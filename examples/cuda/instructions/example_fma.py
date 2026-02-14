"""
This example uses the raw :code:`clock64` hardware counter and ``ncu`` to determine the number of elapsed cycles
per ``FMA`` instruction under two configurations:

1. :py:attr:`Case.SINGLE_THREAD`
2. :py:attr:`Case.MANY_THREADS`

In both cases, the kernel consists in a single **unrolled** *for* loop that performs **dependent** ``FMA`` computations,
*i.e.* each iteration of the *for* loop depends on the result of the ``FMA`` operation of the previous iteration.

Measurements with a single thread and :code:`clock64`
-----------------------------------------------------

Using :py:attr:`Case.SINGLE_THREAD_CLOCK`, *i.e* reading :code:`clock64` before doing the work, and reading it right after,
one can retrieve the value of 4 elapsed cycles per ``FMA`` instruction (see :py:meth:`TestNCU.test_single_thread__clock_avg_cycles_per_fma_instruction_executed`).

Thus, the number of elapsed cycles is about 4 times the number of ``FMA`` instructions.
This means that 4 cycles are needed for the result of the ``FMA`` instruction to be available.
This is further confirmed by :py:meth:`TestSASS.test_stall_of_dependent_is_4`.

Measurements with a single thread and ``ncu``
---------------------------------------------

For the :py:attr:`Case.SINGLE_THREAD`, the number of elapsed cycles per ``FMA`` is also expected to tend to 4, though
it should yield a value larger than with :py:attr:`Case.SINGLE_THREAD_CLOCK` due to ``ncu`` overhead.

Moreover, ``ncu`` reports that it takes on average about 2 active cycles of the ``FMA`` pipeline to complete
a single ``FMA`` instruction, as given by the reading of the `smsp__pipe_fma_cycles_active.sum` metric counter.

These findings are in agreement with :cite:`jia-2019-dissecting-nvidia-turing-gpu` (Table 4.1), in which they states that the dependent-issue latency
for core ``FMA`` math operations is 4 clock cycles for both
:py:attr:`reprospect.tools.architecture.NVIDIAFamily.TURING`
and
:py:attr:`reprospect.tools.architecture.NVIDIAFamily.VOLTA`.

These findings are also in agreement with :cite:`huerta-2025-analyzing-modern-nvidia-gpu` (Section 5.1.1).
In particular, the authors report that fixed-latency instructions have two intermediate stages between the issue stage
and the stage(s) for reading source operands.
The authors call these intermediate stages the "control" and "allocate" stages.

.. table:: ``FMA`` instruction pipeline stages :cite:`huerta-2025-analyzing-modern-nvidia-gpu`
   :align: center

   +-------+-------------+------------------+
   | Cycle | Instruction | Pipeline stage   |
   +=======+=============+==================+
   | 0     | fma         | wait "control"   |
   +-------+-------------+------------------+
   | 1     |             | wait "allocate"  |
   +-------+-------------+------------------+
   | 2     |             | active           |
   +-------+-------------+------------------+
   | 3     |             | active           |
   +-------+-------------+------------------+
   | 4     | fma         | wait "control"   |
   +-------+-------------+------------------+
   | 5     |             | wait "allocate"  |
   +-------+-------------+------------------+
   | 6     |             | active           |
   +-------+-------------+------------------+
   | 7     |             | active           |
   +-------+-------------+------------------+

Measurements with many threads and ``ncu``
------------------------------------------

When launching the same work as for the previous cases, but with a large number of warps, the GPU
is able to hide the latency of instructions by scheduling other warps.

In this case, ``ncu`` reports a number of elapsed cycles that is much closer to the number of ``FMA`` instructions.
Namely, instead of observing 4 elapsed cycles per ``FMA``, it tends to 1 elapsed cycle per ``FMA``, see
:py:meth:`TestNCU.test_many_threads_tends_to_one_active_cycle_per_fma`.

In addition, the ``FMA`` instructions are now distributed to both ``lite`` and ``heavy`` pipelines, see
:py:meth:`TestNCU.test_fma_heavy_and_lite_usage`.
"""

import enum
import logging
import pathlib
import re
import subprocess
import sys
import typing

import pytest

from reprospect.test.case import CMakeAwareTestCase
from reprospect.test.sass.composite import instruction_count_is
from reprospect.test.sass.controlflow.block import BasicBlockMatcher
from reprospect.test.sass.instruction import OpcodeModsMatcher
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.ncu import (
    Cacher,
    Command,
    MetricCorrelation,
    MetricCounter,
    MetricCounterRollUp,
    ProfilingMetrics,
    Report,
)
from reprospect.tools.sass import ControlFlow, Decoder
from reprospect.tools.sass.controlflow import Graph
from reprospect.utils import detect

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Case(enum.IntEnum):
    SINGLE_THREAD_CLOCK = 0
    SINGLE_THREAD = 1
    MANY_THREADS = 2

class TestFMA(CMakeAwareTestCase):
    NITERS: typing.Final[int] = 128 << 4

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_cuda_instructions_fma'

class TestSASS(TestFMA):
    """
    Binary analysis.
    """
    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'examples_cuda_instructions_fma.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope='class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file=self.executable, arch=self.arch, sass=True, cwd=self.cwd, cubin=self.cubin.name, demangler=self.demangler)[0]

    @pytest.fixture(scope='class')
    def decoder(self, cuobjdump: CuObjDump) -> Decoder:
        """
        Get the decoded SASS for :py:attr:`Case.SINGLE_THREAD_CLOCK`.
        """
        [signature] = (signature for signature in cuobjdump.functions if Case.SINGLE_THREAD_CLOCK.name.lower() in signature)
        logging.info(f'Decoding function {signature}.')
        decoder = Decoder(code=cuobjdump.functions[signature].code)

        # TODO: Remove this log.
        logging.info(decoder)

        return decoder

    @pytest.fixture(scope='class')
    def cfg(self, decoder: Decoder) -> Graph:
        cfg = ControlFlow.analyze(instructions=decoder.instructions)
        logging.info(f'There are {len(cfg.blocks)} basic blocks.')
        return cfg

    def test_unrolling(self, cfg: Graph) -> None:
        """
        There are 2 basic blocks with ``FMA`` instructions: one used when the number of iterations is exactly the unrolling value, another one as a fallback.
        """
        fallback, _ = BasicBlockMatcher(matcher=instruction_count_is(
            matcher=OpcodeModsMatcher(opcode='FFMA', operands=True),
            count=1,
        )).assert_matches(cfg=cfg)

        unrolled, _ = BasicBlockMatcher(matcher=instruction_count_is(
            matcher=OpcodeModsMatcher(opcode='FFMA', operands=True),
            count=self.NITERS,
        )).assert_matches(cfg=cfg)

        assert fallback != unrolled

    def test_stall_of_dependent_is_4(self, cfg: Graph) -> None:
        """
        All ``FMA`` instructions of the basic block for the unrolling have a stall count set to 4, except the first and last ones.
        """
        unrolled, _ = BasicBlockMatcher(matcher=instruction_count_is(
            matcher=OpcodeModsMatcher(opcode='FFMA', operands=True),
            count=self.NITERS,
        )).assert_matches(cfg=cfg)

        count_fma = 0
        count_stall_4 = 0
        for instruction in unrolled.instructions:
            if instruction.instruction.startswith('FFMA'):
                count_fma += 1
                if instruction.control.stall_count == 4:
                    count_stall_4 += 1

        assert count_fma == self.NITERS and count_stall_4 == self.NITERS - 2

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestNCU(TestFMA):
    """
    Kernel profiling.
    """
    METRICS: tuple[MetricCounter | MetricCorrelation, ...] = (
        MetricCounter(name='smsp__inst_executed', subs=(MetricCounterRollUp.SUM,)),
        MetricCounter(name='smsp__pipe_fma_cycles_active', subs=(MetricCounterRollUp.SUM, MetricCounterRollUp.AVG, MetricCounterRollUp.MIN, MetricCounterRollUp.MAX)),
        MetricCounter(name='smsp__inst_executed_pipe_fma', subs=(MetricCounterRollUp.SUM, MetricCounterRollUp.AVG, MetricCounterRollUp.MIN, MetricCounterRollUp.MAX)),
        MetricCounter(name='smsp__cycles_elapsed', subs=(MetricCounterRollUp.SUM, MetricCounterRollUp.AVG, MetricCounterRollUp.MIN, MetricCounterRollUp.MAX)),
        MetricCorrelation(name='sass__inst_executed_per_opcode'),
        MetricCounter(name='smsp__inst_executed_pipe_alu', subs=(MetricCounterRollUp.SUM,)),
    )

    METRICS_HEAVY_AND_LITE: tuple[MetricCounter, ...] = (
        MetricCounter(name='smsp__inst_executed_pipe_fmaheavy', subs=(MetricCounterRollUp.SUM, MetricCounterRollUp.AVG, MetricCounterRollUp.MIN, MetricCounterRollUp.MAX)),
        MetricCounter(name='smsp__inst_executed_pipe_fmalite', subs=(MetricCounterRollUp.SUM, MetricCounterRollUp.AVG, MetricCounterRollUp.MIN, MetricCounterRollUp.MAX)),
        MetricCounter(name='smsp__pipe_fmaheavy_cycles_active', subs=(MetricCounterRollUp.SUM, MetricCounterRollUp.AVG, MetricCounterRollUp.MIN, MetricCounterRollUp.MAX)),
        MetricCounter(name='smsp__pipe_fmalite_cycles_active', subs=(MetricCounterRollUp.SUM, MetricCounterRollUp.AVG, MetricCounterRollUp.MIN, MetricCounterRollUp.MAX)),
    )
    """
    `fmaheavy` and `fmalite` pipelines are not available on some architectures.

    See also :py:meth:`available_metrics`.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines
    """

    NVTX_INCLUDES: typing.Final[tuple[str, ...]] = ('fma',)

    def skipif(self) -> None:
        if self.arch.compute_capability == 70:
            pytest.skip(f'{self.arch} does not produce lite and heavy FMA pipeline data.')

    def available_metrics(self) -> tuple[MetricCounter | MetricCorrelation, ...]:
        if self.arch.compute_capability == 70:
            return self.METRICS
        else:
            return self.METRICS + self.METRICS_HEAVY_AND_LITE

    @pytest.fixture(scope='class')
    def cycles(self) -> tuple[int, int]:
        """
        Launch the executable without ``ncu`` to retrieve the measured elapsed cycles with :code:`clock64`.
        """
        logging.info(f'Launching {self.executable} to retrieve raw clock measurements.')
        file = self.cwd / 'clock64.txt'
        subprocess.check_call((self.executable, 'clock', file))
        [niters, cycles] = file.read_text().splitlines()
        return (
            int(re.match(pattern=r'niters: (\d+)', string=niters).group(1)),
            int(re.match(pattern=r'cycles: (\d+)', string=cycles).group(1)),
        )

    @pytest.fixture(scope='class')
    def report(self) -> Report:
        with Cacher() as cacher:
            command = Command(
                output=self.cwd / 'ncu',
                executable=self.executable,
                metrics=self.available_metrics(),
                nvtx_includes=self.NVTX_INCLUDES,
                args=('no-clock',),
            )
            cacher.run(command=command, cwd=self.cwd, retries=5)
        return Report(command=command)

    @pytest.fixture(scope='class')
    def metrics(self, report: Report) -> dict[Case, ProfilingMetrics]:
        results_in_range = report.extract_results_in_range(metrics=self.available_metrics(), includes=self.NVTX_INCLUDES)
        logging.info(results_in_range)
        return {
            Case.SINGLE_THREAD: results_in_range.query_single_next_metrics(accessors=(Case.SINGLE_THREAD.name.lower(),))[1],
            Case.MANY_THREADS:  results_in_range.query_single_next_metrics(accessors=(Case.MANY_THREADS.name.lower(),))[1],
        }

    def test_single_thread_inst_executed_fma(self, metrics: dict[Case, ProfilingMetrics]) -> None:
        assert metrics[Case.SINGLE_THREAD]['sass__inst_executed_per_opcode'].correlated['FFMA'] == self.NITERS

    def test_single_thread_clock64_avg_cycles_per_inst_executed_fma(self, cycles: tuple[int, int]) -> None:
        """
        It tends to 4, but never reaches it.
        """
        cycles_elapsed_per_fma_avg = cycles[1] / cycles[0]
        logging.info(f'Elapsed clock measurements leads to an average of {cycles_elapsed_per_fma_avg} cycles per FFMA.')
        assert cycles_elapsed_per_fma_avg > 4.

    def test_single_thread_avg_cycles_per_inst_executed_fma(self, metrics: dict[Case, ProfilingMetrics]) -> None:
        """
        The `smsp__cycles_elapsed.avg` reading should tend to 4, expect that ``ncu`` has non-negligible overhead.
        """
        smsp__cycles_elapsed_avg = metrics[Case.SINGLE_THREAD]['smsp__cycles_elapsed.avg']
        assert isinstance(smsp__cycles_elapsed_avg, float)
        cycles_elapsed_per_fma_avg = smsp__cycles_elapsed_avg / self.NITERS
        logging.info(f'smsp__cycles_elapsed measurement leads to an average of {cycles_elapsed_per_fma_avg} cycles per FFMA.')
        assert cycles_elapsed_per_fma_avg > 4.

    def test_single_thread_pipe_fma_cycles_active(self, metrics: dict[Case, ProfilingMetrics]) -> None:
        """
        The number of active cycles of the ``FMA`` pipeline is twice the number of unrolled *for* loop iterations in general.
        Some architectures may use instructions such as ``HFMA2`` for other portions of the kernel, thus increasing a bit the
        number of ``FMA`` instructions.
        """
        smsp__inst_executed_pipe_fma_sum = metrics[Case.SINGLE_THREAD]['smsp__inst_executed_pipe_fma.sum']
        smsp__inst_executed_pipe_fma_max = metrics[Case.SINGLE_THREAD]['smsp__inst_executed_pipe_fma.sum']
        smsp__pipe_fma_cycles_active_sum = metrics[Case.SINGLE_THREAD]['smsp__pipe_fma_cycles_active.sum']
        smsp__pipe_fma_cycles_active_max = metrics[Case.SINGLE_THREAD]['smsp__pipe_fma_cycles_active.max']

        assert smsp__inst_executed_pipe_fma_sum == smsp__inst_executed_pipe_fma_max
        assert smsp__pipe_fma_cycles_active_sum == smsp__pipe_fma_cycles_active_max

        assert isinstance(smsp__pipe_fma_cycles_active_max, float)
        assert isinstance(smsp__inst_executed_pipe_fma_max, float)

        # Try to show additional instructions are HFMA2 ?
        if self.arch.compute_capability < 90:
            assert smsp__inst_executed_pipe_fma_max == self.NITERS
            assert smsp__pipe_fma_cycles_active_max == self.NITERS * 2
        else:
            assert smsp__inst_executed_pipe_fma_max == self.NITERS     + 2
            assert smsp__pipe_fma_cycles_active_max == self.NITERS * 2 + 2 * 4

    def test_many_threads_tends_to_one_active_cycle_per_fma(self, metrics: dict[Case, ProfilingMetrics]) -> None:
        smsp__inst_executed_pipe_fma_min = metrics[Case.MANY_THREADS]['smsp__inst_executed_pipe_fma.min']
        smsp__inst_executed_pipe_fma_max = metrics[Case.MANY_THREADS]['smsp__inst_executed_pipe_fma.min']
        smsp__inst_executed_pipe_fma_avg = metrics[Case.MANY_THREADS]['smsp__inst_executed_pipe_fma.avg']

        # The load is perfectly balanced.
        assert smsp__inst_executed_pipe_fma_min == smsp__inst_executed_pipe_fma_max == smsp__inst_executed_pipe_fma_avg

        smsp__cycles_elapsed_avg = metrics[Case.MANY_THREADS]['smsp__cycles_elapsed.avg']

        assert isinstance(smsp__cycles_elapsed_avg, float) and isinstance(smsp__inst_executed_pipe_fma_avg, float)

        cycles_elapsed_per_fma_avg = smsp__cycles_elapsed_avg / smsp__inst_executed_pipe_fma_avg

        logging.info(f'Average number of instructions executed on the FMA pipeline is {smsp__inst_executed_pipe_fma_avg}.')
        logging.info(f'Average number of cycles elapsed is {smsp__cycles_elapsed_avg}.')
        logging.info(f'Average cycles elapsed per FMA instruction is {cycles_elapsed_per_fma_avg}.')

        assert 1.01 < cycles_elapsed_per_fma_avg < 1.2

    def test_fma_heavy_and_lite_usage(self, metrics: dict[Case, ProfilingMetrics]) -> None:
        self.skipif()
        # Only the lite pipeline is used for the single thread case.
        assert metrics[Case.SINGLE_THREAD]['smsp__inst_executed_pipe_fmalite.max'] == self.NITERS
        assert metrics[Case.SINGLE_THREAD]['smsp__inst_executed_pipe_fmalite.min'] == 0
        assert metrics[Case.SINGLE_THREAD]['smsp__inst_executed_pipe_fmaheavy.avg'] == 0

        # Both pipelines are used for the many threads case, and the average active cycle per FMA instruction executed is 2.
        for fmapipe in ('heavy', 'lite'):
            smsp__cycles_active = metrics[Case.MANY_THREADS][f'smsp__pipe_fma{fmapipe}_cycles_active.avg']
            smsp__inst_executed = metrics[Case.MANY_THREADS][f'smsp__inst_executed_pipe_fma{fmapipe}.avg']
            assert isinstance(smsp__cycles_active, float) and isinstance(smsp__inst_executed, float)
            smsp__cycles_elapsed_per_fmasubpipe_instruction_avg = smsp__cycles_active / smsp__inst_executed
            logging.info(f'For FMA pipeline {fmapipe!r}, the number of active cycles per instruction is {smsp__cycles_elapsed_per_fmasubpipe_instruction_avg}.')
            assert smsp__cycles_elapsed_per_fmasubpipe_instruction_avg == 2.
