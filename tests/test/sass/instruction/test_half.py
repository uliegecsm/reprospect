import itertools
import logging
import pathlib
import subprocess
import typing

import pytest

from reprospect.test.features import PTX
from reprospect.test.sass.composite import findall, findunique, instructions_contain, instruction_is, any_of
from reprospect.test.sass.instruction import OpcodeModsMatcher
from reprospect.test.sass.instruction.half import (
    Fp16AddMatcher,
    Fp16FusedMulAddMatcher,
    Fp16MinMaxMatcher,
    Fp16MulMatcher,
)
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass import Decoder

from tests.parameters import Parameters, PARAMETERS
from tests.test.sass.test_instruction import get_decoder

class TestFp16FusedMulAddMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.half.Fp16FusedMulAddMatcher`.
    """
    PTX : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'hfma2.ptx'

    def test_individual(self) -> None:
        matcher = Fp16FusedMulAddMatcher(packed=False)

        assert (matched := matcher.match('HFMA2 R22, R22.H0_H0, R24.H0_H0, R27.H0_H0')) is not None
        assert matched.operands == ('R22', 'R22.H0_H0', 'R24.H0_H0', 'R27.H0_H0')

        assert (matched := matcher.match('HFMA2 R0, R0.H0_H0, R2.H0_H0, 3, 3')) is not None
        assert matched.operands == ('R0', 'R0.H0_H0', 'R2.H0_H0', '3', '3')

    def test_packed(self) -> None:
        matcher = Fp16FusedMulAddMatcher(packed=True)

        assert (matched := matcher.match(inst = 'HFMA2 R7, R2, R2, -RZ')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R7'
        assert matched.operands == ('R7', 'R2', 'R2', '-RZ')

        assert matcher.match('HFMA2 R19, -RZ, RZ, 0, 0') is not None
        assert matcher.match('HFMA2.MMA R57, -RZ, RZ, 0, 0') is not None
        assert matcher.match('HFMA2.MMA R20, -RZ, RZ, 1.875, 0') is not None
        assert matcher.match('HFMA2.MMA R7, R0, 1, 1, R5') is not None

        assert (matched := matcher.match('HFMA2 R24, -RZ, RZ, 0, 1.370906829833984375e-06')) is not None
        assert matched.operands == ('R24', '-RZ', 'RZ', '0', '1.370906829833984375e-06')

    def test_any(self) -> None:
        matcher = Fp16FusedMulAddMatcher(packed=None)

        assert matcher.match('HFMA2 R22, R22.H0_H0, R24.H0_H0, R27.H0_H0') is not None
        assert matcher.match('HFMA2 R19, -RZ, RZ, 0, 0') is not None

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str, scope = 'class')
    def test_from_ptx(self, request, workdir : pathlib.Path, parameters : Parameters) -> None:
        """
        Compile the PTX from :py:attr:`PTX`.
        """
        cubin = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cubin'
        ptx   = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.ptx'

        ptx_isa_version = PTX(arch = parameters.arch).min_isa_version
        ptx.write_text(self.PTX.read_text().format(
            version = f'{ptx_isa_version.major}.{ptx_isa_version.minor}',
            cc = parameters.arch.compute_capability.as_int,
        ))

        subprocess.check_call(('ptxas', '--verbose', f'-arch={parameters.arch.as_sm}', ptx, '-o', cubin))

        cuobjdump = CuObjDump(file = cubin, arch = parameters.arch)

        decoder = Decoder(code = cuobjdump.functions['hfma2'].code)

        matcher = Fp16FusedMulAddMatcher(packed=False)
        matched = tuple(instruction for instruction in decoder.instructions if matcher.match(instruction.instruction))
        counter = sum(1 for instruction in decoder.instructions if instruction.instruction.startswith('HFMA2'))
        assert 2 == len(matched) <= counter

class TestFp16MulMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.half.Fp16MulMatcher`.
    """
    def test_individual(self) -> None:
        assert (matched := Fp16MulMatcher(packed = False).match(inst = 'HMUL2 R0, R2.H0_H0, R3.H0_H0')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands[-1] == 'R3.H0_H0'

    def test_packed(self) -> None:
        assert (matched := Fp16MulMatcher(packed = True).match(inst = 'HMUL2 R0, R2, R3')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands[-1] == 'R3'

    def test_any(self) -> None:
        assert (matched := Fp16MulMatcher().match(inst = 'HMUL2 R0, R2.H1_H1, R3.H0_H0')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands == ('R0', 'R2.H1_H1', 'R3.H0_H0')

        assert (matched := Fp16MulMatcher().match(inst = 'HMUL2 R0, R2, R3')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands == ('R0', 'R2', 'R3')

class TestFp16AddMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.half.Fp16AddMatcher`.
    """
    CODE_AUTO_PACK : typing.Final[str] = """#include "cuda_fp16.h"
__global__ void test_packed(__half* __restrict__ const dst, const __half* __restrict__ const src) {
    dst[0] += src[0]; dst[1] += src[1];
}
"""

    CODE_FORCE_PACK : typing.Final[str] = """#include "cuda_fp16.h"
__global__ void test_packed(__half* __restrict__ const dst, const __half* __restrict__ const src) {
    reinterpret_cast<__half2*>(dst)[0] += reinterpret_cast<const __half2*>(src)[0];
}
"""

    CODE_FP16_TO_FP32 : typing.Final[str] = """#include "cuda_fp16.h"
__global__ void test_fp16_to_fp32(float* __restrict__ const dst, const __half* __restrict__ const src, const unsigned int size)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        dst[index] = __half2float(src[index]);
}
"""

    def test_individual(self) -> None:
        assert (matched := Fp16AddMatcher(packed=False).match(inst = 'HADD2.F32 R6, R2.H0_H0, -RZ.H0_H0')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R6'
        assert matched.operands == ('R6', 'R2.H0_H0', '-RZ.H0_H0')

    def test_packed(self) -> None:
        assert (matched := Fp16AddMatcher(packed = True).match(inst = 'HADD2 R0, R2, R3')) is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R0'
        assert matched.operands == ('R0', 'R2', 'R3')

    def test_any(self) -> None:
        matcher = Fp16AddMatcher(packed=None)

        assert matcher.match('HADD2.F32 R6, R2.H0_H0, -RZ.H0_H0') is not None
        assert matcher.match('HADD2.F32 R6, -RZ, R2.H0_H0') is not None

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_no_auto_pack(self, request, workdir : pathlib.Path, parameters : Parameters, cmake_file_api) -> None:
        """
        The compiler never automatically packs in :py:attr:`CODE_AUTO_PACK`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_AUTO_PACK)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        matcher = Fp16AddMatcher(packed=False)
        assert len(findall(matcher, decoder.instructions)) == 2

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_force_pack(self, request, workdir : pathlib.Path, parameters : Parameters, cmake_file_api) -> None:
        """
        One can force the packing in :py:attr:`CODE_FORCE_PACK`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_FORCE_PACK)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        logging.info(findunique(any_of(
            Fp16AddMatcher(packed=True),
            Fp16FusedMulAddMatcher(packed=True),
        ), decoder.instructions))

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_fp16_to_fp32(self, request, workdir : pathlib.Path, parameters : Parameters, cmake_file_api) -> None:
        """
        Use :py:attr:`CODE_FP16_TO_FP32` to check that the conversion of :code:`__half` to :code:`float`
        always goes through a ``HADD2.F32`` instruction.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_FP16_TO_FP32)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        if parameters.arch.compute_capability >= 80:
            matcher = instruction_is(Fp16AddMatcher()).with_operand(index = 1, operand = '-RZ')
        else:
            matcher = instruction_is(Fp16AddMatcher()).with_operand(index = -1, operand = '-RZ.H0_H0')

        logging.info(findunique(matcher.with_modifier('F32'), instructions=decoder.instructions))

class TestFp16MinMaxMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.half.Fp16MinMaxMatcher`.
    """
    CODE_HMNMX : typing.Final[str] = """\
#include "cuda_fp16.h"

__global__ void test_h{which}(__half* __restrict__ const out, const __half* __restrict__ const src_a, const __half* __restrict__ const src_b, const unsigned int size)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        out[index] = __h{which}(src_a[index], src_b[index]);
}}
"""

    def test_min(self) -> None:
        matcher = Fp16MinMaxMatcher(pmax=False)

        assert (matched := matcher.match('HMNMX2 R5, R2.H0_H0, R5.H0_H0, PT')) is not None
        assert matched.operands == ('R5', 'R2.H0_H0', 'R5.H0_H0', 'PT')

    def test_max(self) -> None:
        matcher = Fp16MinMaxMatcher(pmax=True)

        assert (matched := matcher.match('HMNMX2 R5, R2.H0_H0, R5.H0_H0, !PT')) is not None
        assert matched.operands == ('R5', 'R2.H0_H0', 'R5.H0_H0', '!PT')

    def test_any(self) -> None:
        matcher = Fp16MinMaxMatcher(pmax=None)

        assert matcher.match('HMNMX2 R51, R2.H0_H0, R6.H0_H0, PT') is not None
        assert matcher.match('HMNMX2 R56, R4.H0_H0, R5.H0_H0, !PT') is not None

    @pytest.mark.parametrize(
        ('which', 'parameters'),
        itertools.product(('min', 'max'), PARAMETERS),
        ids = str,
    )
    def test_from_object(self, request, workdir : pathlib.Path, parameters : Parameters, which : typing.Literal['min', 'max'], cmake_file_api) -> None:
        """
        Inspect the behavior of :code:`__hmin` and :code:`__hmax` intrinsics with :py:attr:`CODE_HMNMX`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.{which}.cu'
        FILE.write_text(self.CODE_HMNMX.format(which=which))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        if parameters.arch.compute_capability < 80:
            instructions_contain(OpcodeModsMatcher(opcode='FMNMX')).assert_matches(instructions=decoder.instructions)
        else:
            instructions_contain(Fp16MinMaxMatcher(pmax=which=='max')).assert_matches(instructions=decoder.instructions)
