import math
import pathlib
import re
import subprocess
import typing

import pytest

from reprospect.test.sass.composite import instructions_contain, any_of
from reprospect.test.sass.instruction import LoadGlobalMatcher, StoreGlobalMatcher
from reprospect.test.sass.instruction.half import Fp16MulMatcher, Fp16FusedMulAddMatcher
from reprospect.tools import ncu
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass import ControlFlow, Decoder
from reprospect.tools.sass.controlflow import BasicBlock
from reprospect.utils import cmake, detect

from tests.python.parameters import Parameters, PARAMETERS
from tests.python.test.sass.test_instruction import get_compilation_output

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str, scope = 'class')
class TestSASS:
    """
    Tests that combine different half-precision SASS instructions.
    """
    FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent.parent.parent / 'tests' / 'cpp' / 'cuda' / 'test_half.cu'

    @pytest.fixture(scope = 'class')
    def cuobjdump(self, workdir : pathlib.Path, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> CuObjDump:
        output, _ = get_compilation_output(
            source = self.FILE,
            cwd = workdir,
            arch = parameters.arch,
            object_file = True,
            resource_usage = False,
            cmake_file_api = cmake_file_api,
            ptx = True,
        )

        cuobjdump = CuObjDump(file = output, arch = parameters.arch, sass = True)

        assert len(cuobjdump.functions) == 2

        return cuobjdump

    def test_individual(self, parameters : Parameters, cuobjdump : CuObjDump) -> None:
        """
        Analyse the individual implementation.

        It loads *only* 16 bits at once and does a "broadcast" of the lower lane (``H0_H0``)
        because ``HMUL2`` works with 2 lanes (packed instruction).

        Typically::

            LDG.E.U16.CONSTANT.SYS R2, [R2]
            HMUL2 R0, R2.H0_H0, R2.H0_H0
            STG.E.U16.SYS [R4], R0
        """
        decoder = Decoder(code = cuobjdump.functions['pow2_individual(__half *, const __half *, unsigned int)'].code)
        matcher_load = instructions_contain(LoadGlobalMatcher(arch = parameters.arch, size = 16, readonly = True, extend = 'U'))
        matcher_load.assert_matches(instructions = decoder.instructions)
        matcher_hmul = instructions_contain(Fp16MulMatcher(packed=False))
        matcher_hmul.assert_matches(instructions = decoder.instructions[matcher_load.index:])
        matcher_store = instructions_contain(StoreGlobalMatcher(arch = parameters.arch, size = 16, extend = 'U'))
        matcher_store.assert_matches(instructions = decoder.instructions[matcher_hmul.index:])

    def test_packed(self, parameters : Parameters, cuobjdump : CuObjDump) -> None:
        """
        Analyse the packed implementation.

        First, there is a block that performs the "odd" element and therefore looks like the individual implementation::

          LDG.E.U16.CONSTANT.SYS R2, [R2]
          HMUL2 R0, R2.H0_H0, R2.H0_H0

        Then, there is another block that performs the packed multiplication.
        It loads 32 bits at once.
        Typically::

            LDG.E.CONSTANT.SYS R2, [R2]
            HMUL2 R7, R2, R2

        Note that, even though the PTX always reads::

            mul.f16x2 %r8,%r9,%r9

        ``ptxas`` *may* choose to use ``HFMA2``::

            HFMA2 R7, R2, R2, -RZ

        or even ``HFMA2.MMA``::

            HFMA2.MMA R7, R2, R2, -RZ

        instead of ``HMUL2``, depending on the targeted architecture.
        """
        decoder = Decoder(code = cuobjdump.functions['pow2_packed(__half *, const __half *, unsigned int)'].code)

        cfg = ControlFlow.analyze(instructions = decoder.instructions)

        matcher_load_16 = instructions_contain(LoadGlobalMatcher(arch = parameters.arch, size = 16, readonly = True, extend = 'U'))
        matcher_load_32 = instructions_contain(LoadGlobalMatcher(arch = parameters.arch, size = 32, readonly = True))

        block_individual: BasicBlock | None = None
        block_packed: BasicBlock | None = None

        for block in cfg.blocks:
            if block_individual is None and matcher_load_16.match(instructions = block.instructions) is not None:
                block_individual = block
            elif block_packed is None and matcher_load_32.match(instructions = block.instructions) is not None:
                block_packed = block
            if block_individual is not None and block_packed is not None:
                break

        assert block_individual is not None
        assert block_packed is not None, matcher_load_32.matcher.matchers[0]

        instructions_contain(Fp16MulMatcher(packed = False)).assert_matches(instructions = block_individual.instructions[matcher_load_16.index:])

        instructions_contain(any_of(
            Fp16MulMatcher(packed=True),
            Fp16FusedMulAddMatcher(packed=True),
        )).assert_matches(instructions = block_packed.instructions[matcher_load_32.index:])

        # Let's check that the PTX is actually the same for every architecture.
        ptx = subprocess.check_output(('cuobjdump', '--dump-ptx', cuobjdump.file)).decode()
        assert re.search(r'mul\.f16x2 %r\d+,%r\d+,%r\d+;',  ptx) is not None
        assert re.search(r'mul\.f16 %rs\d+,%rs\d+,%rs\d+;', ptx) is not None

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestNCU:
    """
    `ncu`-based analysis of the individual *vs* packed implementation.
    """
    HALF : typing.Final[pathlib.Path] = pathlib.Path('tests') / 'cpp' / 'cuda' / 'tests_cpp_cuda_half'

    METRICS : typing.Final[tuple[ncu.metrics.MetricKind, ...]] = (
        ncu.MetricDeviceAttribute(name = 'display_name'),
        ncu.L1TEXCacheGlobalLoadInstructions.create(),
        ncu.L1TEXCacheGlobalLoadRequests.create(),
        ncu.L1TEXCacheGlobalLoadSectors.create(),
        *ncu.LaunchGrid.create(),
        *ncu.LaunchBlock.create(),
    )

    WARP_SIZE : typing.Final[int] = 32

    SIZE : typing.Final[int] = 129
    """Buffer size."""

    SIZEOF : typing.Final[int] = 2
    """Size of :code:`__half` in bytes."""

    BLOCK_DIM_X : typing.Final[dict[str, int]] = {
        'individual' : SIZE,
        'packed' : 65,
    }

    @pytest.fixture(scope = 'class')
    def results(self, workdir : pathlib.Path, bindir : pathlib.Path) -> ncu.ProfilingResults:
        with ncu.Cacher(directory = workdir) as cacher:
            command = ncu.Command(
                output = workdir / 'report-half',
                executable = bindir / self.HALF,
                metrics = self.METRICS,
                nvtx_includes = tuple(f'half@{x}/' for x in ('individual', 'packed')),
            )
            cacher.run(command = command, retries = 5)

            report = ncu.Report(command = command)

            return report.extract_results_in_range(metrics = self.METRICS)

    def test_memory(self, results : ncu.ProfilingResults) -> None:
        """
        Compare the memory traffic.
        """
        _, individual = results.query_single_next_metrics(('individual',))
        _, packed     = results.query_single_next_metrics(('packed',))

        for dim in ('x', 'y', 'z'):
            assert individual[f'launch__grid_dim_{dim}'] == 1
            assert packed    [f'launch__grid_dim_{dim}'] == 1

        assert individual['launch__block_dim_y'] == 1 and packed['launch__block_dim_y'] == 1
        assert individual['launch__block_dim_z'] == 1 and packed['launch__block_dim_z'] == 1

        assert individual['launch__block_dim_x'] == self.BLOCK_DIM_X['individual']
        assert packed    ['launch__block_dim_x'] == self.BLOCK_DIM_X['packed']

        assert individual['L1/TEX cache global load instructions sass.sum'] == math.ceil(self.BLOCK_DIM_X['individual'] / self.WARP_SIZE)
        assert packed    ['L1/TEX cache global load instructions sass.sum'] == math.ceil(self.BLOCK_DIM_X['packed']     / self.WARP_SIZE)

        assert individual['L1/TEX cache global load requests.sum'] == math.ceil(self.BLOCK_DIM_X['individual'] / self.WARP_SIZE)
        assert packed    ['L1/TEX cache global load requests.sum'] == math.ceil(self.BLOCK_DIM_X['packed']     / self.WARP_SIZE)

        sectors = math.ceil(self.SIZE * self.SIZEOF / self.WARP_SIZE)

        assert individual['L1/TEX cache global load sectors.sum'] == sectors
        assert packed    ['L1/TEX cache global load sectors.sum'] == sectors
