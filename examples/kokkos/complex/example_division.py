import logging
import pathlib
import re
import sys
import typing

import numpy
import pytest

from reprospect.test import environment
from reprospect.test.case import CMakeAwareTestCase
from reprospect.test.sass.composite import findall
from reprospect.test.sass.instruction import OpcodeModsMatcher
from reprospect.test.sass.instruction.register import Register
from reprospect.test.sass.matchers.convert_fp_to_int import ConvertFpToInt
from reprospect.test.sass.matchers.convert_int_to_fp import ConvertIntToFp
from reprospect.tools.binaries import (
    CuObjDump,
    DetailedRegisterUsage,
    Function,
    NVDisasm,
)
from reprospect.tools.ncu import (
    Cacher,
    Command,
    Metric,
    MetricCorrelation,
    MetricCounter,
    MetricCounterRollUp,
    ProfilingMetrics,
    Report,
)
from reprospect.tools.sass import Decoder
from reprospect.tools.sass.decode import RegisterType
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
    LogbScalbn = 'LogbScalbn'
    ILogbScalbn = 'ILogbScalbn'
    Scaling = 'Scaling'

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
        Method.LogbScalbn:  re.compile(r'reprospect::examples::kokkos::complex::DivisorLogbScalbn<(\(bool\)1|true), (\(bool\)0|false)>'),
        Method.ILogbScalbn: re.compile(r'reprospect::examples::kokkos::complex::DivisorLogbScalbn<(\(bool\)1|true), (\(bool\)1|true)>'),
        Method.Scaling:     re.compile(r'reprospect::examples::kokkos::complex::DivisorScaling<(\(bool\)1|true)>'),
    }

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'{self.get_target_name()}.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope='class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(file=self.executable, arch=self.arch, sass=True, cwd=self.cwd, cubin=self.cubin.name, demangler=self.demangler)[0]

    @pytest.fixture(scope='class')
    def nvdisasm(self, cuobjdump: CuObjDump) -> NVDisasm:
        return NVDisasm(file=cuobjdump.file, arch=self.arch)

    @pytest.fixture(scope='class')
    def function(self, cuobjdump: CuObjDump) -> dict[Method, Function]:
        logging.info(list(cuobjdump.functions.keys()))
        def get_function(method: Method) -> Function:
            pattern = self.SIGNATURE[method]
            return cuobjdump.functions[next(sig for sig in cuobjdump.functions if pattern.search(sig) is not None)]
        return {method: get_function(method) for method in Method}

    @pytest.fixture(scope='class')
    def decoder(self, function: dict[Method, Function]) -> dict[Method, Decoder]:
        return {method: Decoder(code=function[method].code) for method in Method}

    @pytest.fixture(scope='class')
    def detailed_register_usage(self, function: dict[Method, Function], nvdisasm: NVDisasm) -> dict[Method, DetailedRegisterUsage]:
        def get_registers(method: Method) -> DetailedRegisterUsage:
            assert (symbol := function[method].symbol) is not None
            nvdisasm.extract_register_usage_from_liveness_range_info(mangled=(symbol,))
            assert (registers := nvdisasm.functions[symbol].registers) is not None
            return registers
        return {method: get_registers(method) for method in Method}

    def test_division_uses_newton_raphson(self, decoder: dict[Method, Decoder]) -> None:
        """
        Division is implemented using a Newton-Raphson-like method. It starts by computing an approximate
        of the reciprocal of the denominator using the `MUFU.RCP64H` instruction.

        References:

        * https://fp32.org/newton_raphson_division.html
        """
        matcher_mufu = OpcodeModsMatcher(opcode='MUFU', modifiers=('RCP64H',))
        expt_mufu_scaling = 7 if self.arch.compute_capability < 100 else 9
        assert len(findall(matcher=matcher_mufu, instructions=decoder[Method.Scaling].instructions)) == expt_mufu_scaling
        assert len(findall(matcher=matcher_mufu, instructions=decoder[Method.LogbScalbn].instructions)) == 3
        assert len(findall(matcher=matcher_mufu, instructions=decoder[Method.ILogbScalbn].instructions)) == 3

    def test_i2f_f2i_instructions(self, decoder: dict[Method, Decoder]) -> None:
        """
        All methods need to execute 4 INT64 to FP64 conversion instructions to convert the :code:`clock64()` reading to FP64.

        :py:attr:`Method.LogbScalbn` needs additional INT32 to FP64 conversions, likely inside :code:`Kokkos::logb`, as well as
        one `F2I.F64` instruction for :code:`static_cast<int>(logbw)`.
        """
        src_reg = Register.UREG if self.arch.compute_capability >= 120 else Register.REG

        matcher_i2f_64_to_64 = ConvertIntToFp(arch=self.arch, src_dtype=numpy.int64, dst_dtype=numpy.float64, src=src_reg)
        assert len(findall(matcher=matcher_i2f_64_to_64, instructions=decoder[Method.ILogbScalbn].instructions)) == 4
        assert len(findall(matcher=matcher_i2f_64_to_64, instructions=decoder[Method.Scaling].instructions)) == 4
        assert len(findall(matcher=matcher_i2f_64_to_64, instructions=decoder[Method.LogbScalbn].instructions)) == 4

        matcher_i2f_32_to_64 = ConvertIntToFp(arch=self.arch, dst_dtype=numpy.float64, src_dtype=numpy.int32)
        assert len(findall(matcher=matcher_i2f_32_to_64, instructions=decoder[Method.ILogbScalbn].instructions)) == 0
        assert len(findall(matcher=matcher_i2f_32_to_64, instructions=decoder[Method.Scaling].instructions)) == 0
        assert len(findall(matcher=matcher_i2f_32_to_64, instructions=decoder[Method.LogbScalbn].instructions)) > 0

        matcher_f2i_64_to_32 = ConvertFpToInt(src_dtype=numpy.float64, dst_dtype=numpy.int32)
        assert len(findall(matcher=matcher_f2i_64_to_32, instructions=decoder[Method.ILogbScalbn].instructions)) == 0
        assert len(findall(matcher=matcher_f2i_64_to_32, instructions=decoder[Method.Scaling].instructions)) == 0
        assert len(findall(matcher=matcher_f2i_64_to_32, instructions=decoder[Method.LogbScalbn].instructions)) == 1

    def test_detailed_register_usage(self, detailed_register_usage: dict[Method, DetailedRegisterUsage]) -> None:
        match self.arch.compute_capability.as_int:
            case 70:
                expt_ilogbscalbn = {RegisterType.GPR: (40, 36), RegisterType.PRED: (4, 4)}
                expt_logbscalbn =  {RegisterType.GPR: (42, 38), RegisterType.PRED: (4, 4)}
                expt_scaling =     {RegisterType.GPR: (40, 34), RegisterType.PRED: (5, 5)}
            case 75:
                expt_ilogbscalbn = {RegisterType.GPR: (40, 32), RegisterType.PRED: (4, 4), RegisterType.UGPR: (6, 2)}
                expt_logbscalbn =  {RegisterType.GPR: (41, 33), RegisterType.PRED: (4, 4), RegisterType.UGPR: (6, 2)}
                expt_scaling =     {RegisterType.GPR: (40, 32), RegisterType.PRED: (5, 5), RegisterType.UGPR: (6, 2)}
            case 80:
                expt_ilogbscalbn = {RegisterType.GPR: (40, 30), RegisterType.PRED: (4, 4), RegisterType.UGPR: (8, 4)}
                expt_logbscalbn =  {RegisterType.GPR: (40, 32), RegisterType.PRED: (4, 4), RegisterType.UGPR: (8, 4)}
                expt_scaling =     {RegisterType.GPR: (40, 30), RegisterType.PRED: (5, 5), RegisterType.UGPR: (8, 4)}
            case 86:
                expt_ilogbscalbn = {RegisterType.GPR: (42, 38), RegisterType.PRED: (4, 4), RegisterType.UGPR: (8, 4)}
                expt_logbscalbn =  {RegisterType.GPR: (45, 40), RegisterType.PRED: (4, 4), RegisterType.UGPR: (8, 4)}
                expt_scaling =     {RegisterType.GPR: (39, 30), RegisterType.PRED: (5, 5), RegisterType.UGPR: (8, 4)}
            case 89:
                expt_ilogbscalbn = {RegisterType.GPR: (40, 32), RegisterType.PRED: (4, 4), RegisterType.UGPR: (8, 4)}
                expt_logbscalbn =  {RegisterType.GPR: (40, 32), RegisterType.PRED: (4, 4), RegisterType.UGPR: (8, 4)}
                expt_scaling =     {RegisterType.GPR: (39, 30), RegisterType.PRED: (5, 5), RegisterType.UGPR: (8, 4)}
            case 90:
                expt_ilogbscalbn = {RegisterType.GPR: (40, 32), RegisterType.PRED: (4, 4), RegisterType.UGPR: (9, 5)}
                expt_logbscalbn =  {RegisterType.GPR: (40, 34), RegisterType.PRED: (4, 4), RegisterType.UGPR: (9, 5)}
                expt_scaling =     {RegisterType.GPR: (40, 33), RegisterType.PRED: (5, 5), RegisterType.UGPR: (9, 5)}
            case 100:
                expt_ilogbscalbn = {RegisterType.GPR: (45, 40), RegisterType.PRED: (4, 4), RegisterType.UGPR: (9, 5)}
                expt_logbscalbn =  {RegisterType.GPR: (45, 40), RegisterType.PRED: (4, 4), RegisterType.UGPR: (9, 5)}
                expt_scaling =     {RegisterType.GPR: (40, 34), RegisterType.PRED: (4, 4), RegisterType.UGPR: (10, 6)}
            case 120:
                expt_ilogbscalbn = {RegisterType.GPR: (45, 39), RegisterType.PRED: (4, 4), RegisterType.UGPR: (10, 6)}
                expt_logbscalbn =  {RegisterType.GPR: (45, 40), RegisterType.PRED: (4, 4), RegisterType.UGPR: (10, 6)}
                expt_scaling =     {RegisterType.GPR: (40, 32), RegisterType.PRED: (4, 4), RegisterType.UGPR: (10, 6)}
            case _:
                raise ValueError(f'unsupported {self.arch}')

        assert detailed_register_usage[Method.ILogbScalbn] == expt_ilogbscalbn
        assert detailed_register_usage[Method.LogbScalbn] == expt_logbscalbn
        assert detailed_register_usage[Method.Scaling] == expt_scaling

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestNCU(TestDivision):
    """
    Kernel profiling.
    """
    METRICS: tuple[Metric | MetricCorrelation, ...] = (
        MetricCounter(name='smsp__inst_executed', subs=(MetricCounterRollUp.SUM,)),
        MetricCorrelation(name='sass__inst_executed_per_opcode'),
        MetricCounter(name='sm__inst_executed_pipe_fp64', subs=(MetricCounterRollUp.SUM,)),
        MetricCounter(name='sm__pipe_fp64_cycles_active', subs=(MetricCounterRollUp.SUM,)),
    )

    NVTX_INCLUDES: typing.Final[tuple[str, ...]] = ('division',)

    ELEMENT_COUNT: typing.Final[int] = 128 << 1
    WARP_SIZE: typing.Final[int] = 32
    WARP_COUNT: typing.Final[int] = ELEMENT_COUNT // WARP_SIZE

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
            cacher.run(command=command, cwd=self.cwd, retries=5)
        return Report(command=command)

    @pytest.fixture(scope='class')
    def metrics(self, report: Report) -> dict[Method, ProfilingMetrics]:
        results_in_range = report.extract_results_in_range(
            metrics=self.METRICS,
            includes=self.NVTX_INCLUDES,
        )

        def get_metrics(method: Method) -> ProfilingMetrics:
            kernel, results = results_in_range.query(accessors=(method.name,)).query_single_next(accessors=())
            _, metrics = results.query_single_next_metrics(accessors=())
            logging.info(f'Kernel {kernel} profiling results for {method!r} method:\n{results}')
            return metrics

        return {method: get_metrics(method) for method in Method}

    def test_mufu(self, metrics: dict[Method, ProfilingMetrics]) -> None:
        """
        The number of executed `MUFU` instructions is equal to the number of divisions performed per work item times :py:attr:`WARP_COUNT`.
        """
        assert metrics[Method.Scaling]['sass__inst_executed_per_opcode'].correlated['MUFU'] == 6 * self.WARP_COUNT
        assert metrics[Method.ILogbScalbn]['sass__inst_executed_per_opcode'].correlated['MUFU'] == 2 * self.WARP_COUNT
        assert metrics[Method.LogbScalbn]['sass__inst_executed_per_opcode'].correlated['MUFU'] == 2 * self.WARP_COUNT

    def test_fp_instructions(self, metrics: dict[Method, ProfilingMetrics]) -> None:
        """
        :py:data:`Method.LogbScalbn` and :py:data:`Method.ILogbScalbn` differ in their `DSETP` count.

        :py:data:`Method.LogbScalbn` calls :code:`Kokkos::isfinite` on the :code:`Kokkos::logb` result, emitting a `DSETP`.
        :py:data:`Method.ILogbScalbn` replaces this with an integer comparison, staying on the integer pipeline.
        """
        for opcode in ('DMUL', 'DFMA', 'FSETP', 'DADD', 'FFMA'):
            assert metrics[Method.LogbScalbn]['sass__inst_executed_per_opcode'].correlated[opcode] \
                == metrics[Method.ILogbScalbn]['sass__inst_executed_per_opcode'].correlated[opcode], opcode

        assert metrics[Method.LogbScalbn]['sass__inst_executed_per_opcode'].correlated['DSETP'] \
             > metrics[Method.ILogbScalbn]['sass__inst_executed_per_opcode'].correlated['DSETP']

        assert metrics[Method.LogbScalbn]['sass__inst_executed_per_opcode'].correlated['ISETP'] \
             < metrics[Method.ILogbScalbn]['sass__inst_executed_per_opcode'].correlated['ISETP']

    def test_i2f_f2i_instructions(self, metrics: dict[Method, ProfilingMetrics]) -> None:
        """
        Confirms :py:meth:`TestSASS.test_i2f_f2i_instructions`.
        """
        assert metrics[Method.ILogbScalbn]['sass__inst_executed_per_opcode'].correlated['I2F'] == 4 * self.WARP_COUNT
        assert metrics[Method.Scaling]['sass__inst_executed_per_opcode'].correlated['I2F'] == 4 * self.WARP_COUNT
        assert metrics[Method.LogbScalbn]['sass__inst_executed_per_opcode'].correlated['I2F'] == 5 * self.WARP_COUNT

        assert 'F2I' not in metrics[Method.ILogbScalbn]['sass__inst_executed_per_opcode'].correlated
        assert 'F2I' not in metrics[Method.Scaling]['sass__inst_executed_per_opcode'].correlated
        assert metrics[Method.LogbScalbn]['sass__inst_executed_per_opcode'].correlated['F2I'] == 1 * self.WARP_COUNT

    def test_fp64_instructions_and_cycles(self, metrics: dict[Method, ProfilingMetrics]) -> None:
        """
        In terms of FP64 instructions executed and active pipeline cycles,
        :py:data:`Method.ILogbScalbn` outperforms :py:data:`Method.LogbScalbn`,
        which in turn outperforms :py:data:`Method.Scaling`.
        """
        fp64_inst: typing.Final[dict[Method, int]] =   {m: typing.cast(int, metrics[m]['sm__inst_executed_pipe_fp64.sum']) for m in Method}
        fp64_cycles: typing.Final[dict[Method, int]] = {m: typing.cast(int, metrics[m]['sm__pipe_fp64_cycles_active.sum']) for m in Method}

        assert fp64_inst[Method.Scaling] >    fp64_inst[Method.LogbScalbn]
        assert fp64_inst[Method.LogbScalbn] > fp64_inst[Method.ILogbScalbn]

        assert fp64_cycles[Method.Scaling] >    fp64_cycles[Method.LogbScalbn]
        assert fp64_cycles[Method.LogbScalbn] > fp64_cycles[Method.ILogbScalbn]
