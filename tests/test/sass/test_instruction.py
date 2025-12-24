import functools
import logging
import os
import pathlib

import pytest
import regex
import semantic_version

from reprospect.test.sass.composite import findunique
from reprospect.test.sass.instruction import (
    AtomicMatcher,
    OpcodeModsMatcher,
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
    ReductionMatcher,
)
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass import Decoder
from reprospect.utils import cmake

from tests.compilation import get_compilation_output
from tests.parameters import PARAMETERS, Parameters

CODE_ELEMENTWISE_ADD_RESTRICT = """\
__global__ void elementwise_add_restrict(int* __restrict__ const dst, const int* __restrict__ const src) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] += src[index];
}
"""
"""Element-wise add with 32-bit :code:`int`."""

CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE = """\
__global__ void elementwise_add_restrict_128_wide(float4* __restrict__ const dst, const float4* __restrict__ const src) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const float4& a = src[index];
    const float4& b = dst[index];
    dst[index] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
"""
"""Element-wise add with 128-bit :code:`float4`."""

CODE_ELEMENTWISE_ADD_RESTRICT_256_WIDE = """\
struct alignas(4 * sizeof(double)) Tester {
    double x, y, z, w;
};

__global__ void elementwise_add_restrict_256_wide(Tester* __restrict__ const dst, const Tester* __restrict__ const src)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const Tester& a = src[index];
    const Tester& b = dst[index];
    dst[index] = Tester{.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w};
}
"""
"""Element-wise add with 256-bit aligned elements."""

@functools.lru_cache(maxsize=128)
def get_decoder(*, cwd: pathlib.Path, arch: NVIDIAArch, file: pathlib.Path, cmake_file_api: cmake.FileAPI, **kwargs) -> tuple[Decoder, pathlib.Path]:
    """
    Compile the code in `file` for `arch` and return a :py:class:`reprospect.tools.sass.Decoder`.
    """
    output, _ = get_compilation_output(
        source=file,
        cwd=cwd,
        arch=arch,
        object_file=True,
        resource_usage=False,
        cmake_file_api=cmake_file_api,
        **kwargs,
    )

    cuobjdump = CuObjDump(file=output, arch=arch, sass=True)

    assert len(cuobjdump.functions) == 1

    return Decoder(code=next(iter(cuobjdump.functions.values())).code), output

class TestPatternBuilder:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.PatternBuilder`.
    """
    def test_reg_vs_ureg(self) -> None:
        """
        Ensure that :py:attr:`reprospect.test.sass.instruction.PatternBuilder.REG` does not
        match what :py:attr:`reprospect.test.sass.instruction.PatternBuilder.UREG` matches (and
        vice versa).
        """
        assert regex.match(PatternBuilder.REG,    'R42') is not None
        assert regex.match(PatternBuilder.reg(),  'R42') is not None
        assert regex.match(PatternBuilder.UREG,   'R42') is None
        assert regex.match(PatternBuilder.ureg(), 'R42') is None

        assert regex.match(PatternBuilder.REG,    'UR42') is None
        assert regex.match(PatternBuilder.reg(),  'UR42') is None
        assert regex.match(PatternBuilder.UREG,   'UR42') is not None
        assert regex.match(PatternBuilder.ureg(), 'UR42') is not None

    def test_anygpreg(self) -> None:
        """
        Test :py:meth:`reprospect.test.sass.instruction.PatternBuilder.anygpreg`.
        """
        assert regex.match(PatternBuilder.anygpreg(reuse=None, group='mygroup'),  'R42').captures('mygroup') == ['R42']
        assert regex.match(PatternBuilder.anygpreg(reuse=None, group='mygroup'), 'UR42').captures('mygroup') == ['UR42']

        assert regex.match(PatternBuilder.anygpreg(reuse=False), 'R66.reuse').group() == 'R66'
        assert regex.match(PatternBuilder.anygpreg(reuse=True),  'R66.reuse').group() == 'R66.reuse'

@pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
class TestReductionMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.ReductionMatcher`.
    """
    CODE_ADD = """\
__global__ void add({type}* __restrict__ const dst, const {type}* __restrict__ const src)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&dst[index], src[index]);
}}
"""

    CODE_SUB = """\
__global__ void sub(int* __restrict__ const dst, const int* __restrict__ const src)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicSub(&dst[index], src[index]);
}
"""

    CODE_MAX = """\
__global__ void max({type}* __restrict__ const dst, const {type}* __restrict__ const src)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicMax(&dst[index], src[index]);
}}
"""

    def test_add_strong_device_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type='int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch=parameters.arch, operation='ADD', scope='DEVICE', consistency='STRONG', dtype=('S', 32))
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(f'{matcher} matched instruction {inst.instruction} as {matched}.')

        assert {'ADD'}.issubset(matched.modifiers)
        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

        # Another consistency would fail.
        matcher = ReductionMatcher(arch=parameters.arch, operation='ADD', scope='DEVICE', consistency='WEAK', dtype=('S', 32))
        assert not any(matcher.match(inst) for inst in decoder.instructions)

    def test_add_strong_device_unsigned_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `unsigned int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type='unsigned int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch=parameters.arch, operation='ADD', scope='DEVICE', consistency='STRONG', dtype=('U', 32))
        matched = findunique(matcher, decoder.instructions)

        assert {'ADD'}.issubset(matched.modifiers)
        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

    def test_add_strong_device_unsigned_long_long_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `unsigned long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type='unsigned long long int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(
            arch=parameters.arch,
            operation='ADD',
            scope='DEVICE',
            consistency='STRONG',
            dtype=('U', 64),
        )
        matched = findunique(matcher, decoder.instructions)

        assert {'ADD', '64'}.issubset(matched.modifiers)
        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

    def test_add_strong_device_float(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `float`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type='float'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch=parameters.arch, operation='ADD', dtype=('F', 32), scope='DEVICE', consistency='STRONG')
        matched = findunique(matcher, decoder.instructions)

        assert {'F32', 'FTZ', 'RN'}.issubset(matched.modifiers)
        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

    def test_add_strong_device_double(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `double`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type='double'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch=parameters.arch, operation='ADD', dtype=('F', 64), scope='DEVICE', consistency='STRONG')
        matched = findunique(matcher, decoder.instructions)

        assert {'F64', 'RN'}.issubset(matched.modifiers)
        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

    def test_sub_strong_device(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_SUB`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_SUB)

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        # Note that the source is negated in another instruction.
        logging.info(findunique(
            matcher=ReductionMatcher(arch=parameters.arch, operation='ADD', scope='DEVICE', consistency='STRONG'),
            instructions=decoder.instructions,
        ))

    def test_max_strong_device_long_long_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `long long int`. The modifier is ``MAX.S64``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type='long long int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher_type: type[AtomicMatcher | ReductionMatcher]
        if semantic_version.Version(os.environ['CUDA_VERSION']) in semantic_version.SimpleSpec('<12.8'):
            matcher_type = AtomicMatcher
        else:
            matcher_type = ReductionMatcher
        matcher = matcher_type(arch=parameters.arch, operation='MAX', dtype=('S', 64), scope='DEVICE', consistency='STRONG')
        matched = findunique(matcher, decoder.instructions)

        assert {'MAX', 'S64'}.issubset(matched.modifiers)

    def test_max_strong_device_unsigned_long_long_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `unsigned long long int`. The modifier is ``MAX.64``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type='unsigned long long int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher_type: type[AtomicMatcher | ReductionMatcher]
        if semantic_version.Version(os.environ['CUDA_VERSION']) in semantic_version.SimpleSpec('<12.8'):
            matcher_type = AtomicMatcher
        else:
            matcher_type = ReductionMatcher
        matcher = matcher_type(arch=parameters.arch, operation='MAX', dtype=('U', 64), scope='DEVICE', consistency='STRONG')
        matched = findunique(matcher, decoder.instructions)

        assert {'MAX', '64'}.issubset(matched.modifiers)

    def test_max_strong_device_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `int`. The modifier is ``MAX.S32``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type='int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch=parameters.arch, operation='MAX', dtype=('S', 32), scope='DEVICE', consistency='STRONG')
        matched = findunique(matcher, decoder.instructions)

        assert {'MAX', 'S32'}.issubset(matched.modifiers)

    def test_max_strong_device_unsigned_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `unsigned int`. The modifier is ``MAX``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type='unsigned int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch=parameters.arch, operation='MAX', dtype=('U', 32), scope='DEVICE', consistency='STRONG')
        matched = findunique(matcher, decoder.instructions)

        assert {'MAX'}.issubset(matched.modifiers)

class TestOpcodeModsMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.OpcodeModsMatcher`.
    """
    def test_with_square_brackets(self):
        instruction = 'IMAD R4, R4, c[0x0][0x0], R3'

        matched = OpcodeModsMatcher(opcode='IMAD').match(instruction)

        assert matched is not None
        assert matched.operands == ('R4', 'R4', 'c[0x0][0x0]', 'R3')

    def test_with_minus_sign(self):
        instruction = 'UIADD3 UR5, UPT, UPT, -UR4, UR9, URZ'

        matched = OpcodeModsMatcher(opcode='UIADD3').match(instruction)

        assert matched is not None
        assert matched.operands == ('UR5', 'UPT', 'UPT', '-UR4', 'UR9', 'URZ')

    def test_with_reuse(self):
        instruction = 'ISETP.GE.U32.AND P0, PT, R0.reuse, UR5, PT'

        matched = OpcodeModsMatcher(opcode='ISETP', modifiers=('GE', 'U32', 'AND')).match(instruction)

        assert matched is not None
        assert matched.operands == ('P0', 'PT', 'R0.reuse', 'UR5', 'PT')

    def test_with_descr(self):
        instruction = 'LDG.E R2, desc[UR6][R2.64]'

        matched = OpcodeModsMatcher(opcode='LDG', modifiers=('E',)).match(instruction)

        assert matched is not None
        assert matched.operands == ('R2', 'desc[UR6][R2.64]')

class TestOpcodeModsWithOperandsMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.OpcodeModsWithOperandsMatcher`.
    """
    def test(self):
        instruction = 'ISETP.NE.AND P2, PT, R4, RZ, PT'

        matcher = OpcodeModsWithOperandsMatcher(opcode='ISETP', modifiers=('NE', 'AND'), operands=(
            PatternBuilder.PRED,
            PatternBuilder.PREDT,
            'R4',
            PatternBuilder.REGZ,
            PatternBuilder.PREDT,
        ))

        assert matcher.pattern.pattern == (
            r'(?P<opcode>ISETP)\.(?P<modifiers>NE)\.(?P<modifiers>AND)\s*'
            r'(?P<operands>P[0-9]+),\s+'
            r'(?P<operands>P(?:T|\d+)),\s+'
            r'(?P<operands>R4),\s+'
            r'(?P<operands>R(?:Z|\d+)),\s+'
            r'(?P<operands>P(?:T|\d+))'
        )

        matched = matcher.match(instruction)

        logging.info(f'{matcher} matched instruction {instruction} as {matched}.')

        assert matched is not None
        assert matched.operands == ('P2', 'PT', 'R4', 'RZ', 'PT')
