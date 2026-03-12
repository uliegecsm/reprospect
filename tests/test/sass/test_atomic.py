import logging
import pathlib
import subprocess
import typing

import numpy
import pytest
import regex
import semantic_version

from reprospect.test.sass.instruction import (
    AtomicMatcher,
    InstructionMatch,
    ThreadScope,
)
from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.sass import Instruction
from reprospect.utils import cmake

from tests.parameters import PARAMETERS, Parameters
from tests.test.sass.test_instruction import get_decoder


@pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
class TestAtomicMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.atomic.AtomicMatcher`.
    """
    CODE_ADD_BASED_ON_CAS = """\
__global__ void cas({type}* __restrict__ const dst, const {type}* __restrict__ const src)
{{
    static_assert(sizeof({type}) == {size});

    const auto index = blockIdx.x * blockDim.x + threadIdx.x;

    {integer}* const dest = reinterpret_cast<{integer}*>(dst + index);

    {integer} old = *dest;
    {integer} assumed;

    do {{
        assumed = old;
        {type} new_val = assumed + src[index];
        old = atomicCAS(
            dest,
            assumed,
            reinterpret_cast<{integer}&>(new_val)
        );
    }} while (old != assumed);
}}
"""

    CODE_ADD_RELAXED_BLOCK = """\
#include "cuda/atomic"

__global__ void add({type}* __restrict__ const dst, const {type}* __restrict__ const src)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;

    cuda::atomic_ref<{type}, cuda::thread_scope_block> ref(dst[index]);
    ref.fetch_add(src[index], cuda::memory_order_relaxed);
}}
"""

    CODE_MIN = """\
#include "cuda/atomic"

__global__ void add({type}* __restrict__ const dst, const {type}* __restrict__ const src)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::atomic_ref<{type}, cuda::thread_scope_{scope}> ref(dst[index]);
    ref.fetch_min(src[index], cuda::memory_order_relaxed);
}}
"""

    CODE_EXCH = """\
__global__ void exch({type}* __restrict__ const dst, const {type}* __restrict__ const src)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicExch(&dst[index], src[index]);
}}
"""

    CODE_COMPARE_EXCHANGE = """\
#include "cuda/atomic"

__global__ void compare_exchange({type}* __restrict__ dst, const {type}* __restrict__ src)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::atomic_ref<{type}, cuda::thread_scope_{scope}> ref(dst[index]);

    {type} old_val = ref.load(cuda::memory_order_relaxed);
    {type} new_val;

    do {{
        new_val = min(old_val, src[index]);
    }} while (!ref.compare_exchange_{consistency}(old_val, new_val,
                                       cuda::memory_order_acquire,
                                       cuda::memory_order_acquire));
}}
"""

    CODE_ADD_BASED_ON_CAS_128 = """\
struct alignas(2 * sizeof(double)) My128Struct
{
    double x, y;

    __host__ __device__ friend My128Struct operator+(const My128Struct& a, const My128Struct& b)
    {
        return My128Struct{
            .x = a.x + b.x,
            .y = a.y + b.y
        };
    }

    auto operator<=>(const My128Struct&) const = default;
};

__global__ void cas(My128Struct* __restrict__ const dst, const My128Struct* __restrict__ const src)
{
    static_assert(sizeof (My128Struct) == 16);
    static_assert(alignof(My128Struct) == 16);

    static_assert(std::is_trivially_copyable_v<My128Struct>);

    const auto index = blockIdx.x * blockDim.x + threadIdx.x;

    auto* const dest = dst + index;

    My128Struct old = *dest;
    My128Struct assumed;

    do {
        assumed = old;
        My128Struct new_val = assumed + src[index];
        old = atomicCAS(
            dest,
            assumed,
            new_val
        );
    } while (old != assumed);
}
"""

    CODE_EXCH_DEVICE_PTR: typing.Final[str] = """\
__device__ __constant__ int32_t* ptr;

__global__ void atomic_exch_kernel() {
    atomicExch(ptr, 0);
}
"""

    @staticmethod
    def match_one(*, decoder, **kwargs) -> tuple[AtomicMatcher, Instruction, InstructionMatch]:
        """
        Match exactly one instruction.
        """
        matcher = AtomicMatcher(**kwargs)
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(red) == 1, matcher

        inst, matched = red[0]

        logging.info(f'{matcher} matched instruction {inst.instruction} as {matched}.')

        return matcher, inst, matched

    @pytest.mark.parametrize(
        'word', [
            ( 16, 'short int', 'unsigned short int'),
            ( 32, 'float',     'unsigned int'),
            ( 64, 'double',    'unsigned long long int'),
    ], ids=str)
    def test_atomicCAS(self, request, workdir, word, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD_BASED_ON_CAS`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}{word[0]}.cu'
        FILE.write_text(self.CODE_ADD_BASED_ON_CAS.format(
            size=int(word[0] / 8),
            type=word[1],
            integer=word[2],
        ))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the atomic CAS.
        if cmake_file_api.toolchains['CUDA']['compiler']['id'] == 'Clang' and \
            semantic_version.Version(cmake_file_api.toolchains['CUDA']['compiler']['version']) in semantic_version.SimpleSpec('>21'):
            expt_scope = 'SYSTEM'
        else:
            expt_scope = 'DEVICE'
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='CAS', consistency='STRONG', scope=expt_scope,
            dtype=word[0],
        )

        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 5
        assert matched.operands[0] == 'PT'

    def test_atomicCAS_128(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Supported from compute capability 9.x.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.128.cu'
        FILE.write_text(self.CODE_ADD_BASED_ON_CAS_128)

        # Under some circumstances, 128-bit atomicCAS will not compile.
        expecting_failure = False
        match cmake_file_api.toolchains['CUDA']['compiler']['id']:
            case 'NVIDIA':
                if parameters.arch.compute_capability < 90:
                    logging.warning('The 128-bit atomicCAS() is only supported by devices of compute capability 9.x and higher.')
                    expecting_failure = True
            case 'Clang':
                logging.warning('The 128-bit atomicCAS() does not compile with Clang.')
                expecting_failure = True
            case _:
                raise ValueError(f"unsupported compiler {cmake_file_api.toolchains['CUDA']['compiler']}")

        kwargs = {'cwd': workdir, 'arch': parameters.arch, 'file': FILE, 'cmake_file_api': cmake_file_api}

        if expecting_failure:
            with pytest.raises(subprocess.CalledProcessError):
                get_decoder(**kwargs)
            return
        decoder, _ = get_decoder(**kwargs)

        # Find the atomic CAS.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='CAS', consistency='STRONG', scope='DEVICE',
            dtype=128,
        )

        assert {'CAS', '128'}.issubset(matched.modifiers)

    def test_add_relaxed_block_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        As of CUDA 13.0.0, the generated code still applies the ``.STRONG`` modifier,
        regardless of the ``.relaxed`` qualifier shown in the PTX.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type='int'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic add.
        matcher, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='ADD',
            consistency='STRONG', scope='BLOCK',
            memory='',
            dtype=numpy.int32,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        if matcher.version in semantic_version.SimpleSpec('<12.8'):
            assert 'atom.add.relaxed.cta.u32' in result
        else:
            assert 'atom.add.relaxed.cta.s32' in result

    def test_add_relaxed_block_unsigned_long_long_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Similar to :py:meth:`test_add_relaxed_block_int` for `unsigned long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type='unsigned long long int'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic add.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='ADD',
            consistency='STRONG', scope='BLOCK',
            memory='',
            dtype=numpy.uint64,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.add.relaxed.cta.u64' in result, result

    def test_add_relaxed_block_float(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Similar to :py:meth:`test_add_relaxed_block_int` for `float`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type='float'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic add.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='ADD',
            consistency='STRONG', scope='BLOCK',
            memory='',
            dtype=numpy.float32,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.add.relaxed.cta.f32' in result

    def test_add_relaxed_block_double(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Similar to :py:meth:`test_add_relaxed_block_int` for `double`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type='double'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic add.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='ADD',
            consistency='STRONG', scope='BLOCK',
            memory='',
            dtype=numpy.float64,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.add.relaxed.cta.f64' in result

    def test_min_relaxed_device_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MIN` for `int` and :py:data:`reprospect.test.sass.instruction.ThreadScope.DEVICE` scope.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MIN.format(type='int', scope='device'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic min.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='MIN',
            consistency='STRONG', scope=ThreadScope.DEVICE,
            memory='',
            dtype=numpy.int32,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'
        assert {'MIN', 'S32'}.issubset(matched.modifiers)

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.min.relaxed.gpu.s32' in result

    def test_min_relaxed_system_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MIN` for `int` and :py:data:`reprospect.test.sass.instruction.ThreadScope.SYSTEM` scope.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MIN.format(type='int', scope='system'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic min.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='MIN',
            consistency='STRONG', scope=ThreadScope.SYSTEM,
            memory='',
            dtype=numpy.int32,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'
        assert {'MIN', 'S32'}.issubset(matched.modifiers)

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.min.relaxed.sys.s32' in result

    def test_min_relaxed_device_long_long_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MIN` for `long long int` and :py:data:`reprospect.test.sass.instruction.ThreadScope.DEVICE` scope.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MIN.format(type='long long int', scope='device'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic min.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='MIN',
            consistency='STRONG', scope='DEVICE',
            memory='',
            dtype=numpy.int64,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'
        assert {'MIN', 'S64'}.issubset(matched.modifiers)

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.min.relaxed.gpu.s64' in result

    def test_min_relaxed_device_unsigned_long_long_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MIN` for `unsigned long long int` and :py:data:`reprospect.test.sass.instruction.ThreadScope.DEVICE` scope.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MIN.format(type='unsigned long long int', scope='device'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic min.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='MIN',
            consistency='STRONG', scope='DEVICE',
            memory='',
            dtype=numpy.uint64,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert matched.operands[1] == 'RZ'
        assert {'MIN', '64'}.issubset(matched.modifiers)

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.min.relaxed.gpu.u64' in result

    @pytest.mark.parametrize('consistency', ['strong', 'weak'], ids=str)
    def test_compare_exchange_system(self, request, consistency: str, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_COMPARE_EXCHANGE` for `unsigned long long int`, :py:data:`reprospect.test.sass.instruction.ThreadScope.SYSTEM` scope.

        .. note::

            Both *weak*  and *strong* lead to a `STRONG` consistency.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_COMPARE_EXCHANGE.format(type='unsigned long long int', consistency=consistency, scope='system'))

        decoder, output = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        # Find the atomic CAS.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch,
            operation='CAS',
            consistency='STRONG', scope=ThreadScope.SYSTEM,
            memory=MemorySpace.GENERIC,
            dtype=numpy.uint64,
        )

        assert regex.match(Register.PREDT, matched.operands[0]) is not None
        assert {'CAS', '64'}.issubset(matched.modifiers)

        # In the PTX, we can see the '.acquire'.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert 'atom.cas.acquire.sys.b64' in result

    def test_exch_strong_device_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_EXCH` for `int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_EXCH.format(type='int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the atomic exchange.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch, operation='EXCH', dtype=numpy.int32, scope='DEVICE', consistency='STRONG',
        )

        assert {'EXCH'}.issubset(matched.modifiers)

    def test_exch_strong_device_unsigned_long_long_int(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_EXCH` for `unsigned long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_EXCH.format(type='unsigned long long int'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the atomic exchange.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch, operation='EXCH', dtype=numpy.uint64, scope='DEVICE', consistency='STRONG',
        )

        assert {'EXCH', '64'}.issubset(matched.modifiers)

    def test_exch_strong_device_float(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:attr:`CODE_EXCH` for `float`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_EXCH.format(type='float'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        # Find the atomic exchange.
        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch, operation='EXCH', dtype=numpy.float32, scope='DEVICE', consistency='STRONG',
        )

        assert {'EXCH'}.issubset(matched.modifiers)

    def test_exch_device_ptr(self, request, workdir: pathlib.Path, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
        """
        This test demonstrates that while ``nvcc`` emits an ``ATOMG`` instruction for an atomic exchange using a device pointer marked with
        :code:`__constant__`, ``clang`` (as of 21.1.5) is not able to infer that the referenced memory resides in global memory and therefore falls back
        emitting a generic ``ATOM`` instruction.

        ``nvcc`` appears to generate better code in this case: because the device pointer is declared as :code:`__constant__`,
        the compiler can reasonably assume that it cannot point to local or shared memory, and thus must refer to global memory.
        This allows ``nvcc`` to use the more specific global-memory atomic instruction.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_EXCH_DEVICE_PTR)

        decoder, output = get_decoder(
            cwd=workdir, arch=parameters.arch, file=FILE,
            cmake_file_api=cmake_file_api, ptx=True,
        )

        # Find the atomic exchange.
        memory: MemorySpace
        match cmake_file_api.toolchains['CUDA']['compiler']['id']:
            case 'NVIDIA':
                memory = MemorySpace.GLOBAL
            case 'Clang':
                memory = MemorySpace.GENERIC
            case _:
                raise ValueError(f"unsupported compiler {cmake_file_api.toolchains['CUDA']['compiler']}")

        _, _, matched = self.match_one(
            decoder=decoder,
            arch=parameters.arch, operation='EXCH', dtype=32, scope='DEVICE', consistency='STRONG',
            memory=memory,
        )

        assert {'EXCH'}.issubset(matched.modifiers)

        # In the PTX, we can see the '.global' for NVIDIA, but not for Clang.
        result = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        match cmake_file_api.toolchains['CUDA']['compiler']['id']:
            case 'NVIDIA':
                assert 'atom.global.exch.b32' in result
            case 'Clang':
                assert 'atom.exch.b32' in result
            case _:
                raise ValueError(f"unsupported compiler {cmake_file_api.toolchains['CUDA']['compiler']}")
