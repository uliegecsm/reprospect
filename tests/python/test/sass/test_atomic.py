import logging
import os
import pathlib
import subprocess

import pytest
import regex
import semantic_version
import typeguard

from reprospect.tools.sass import Instruction
from reprospect.test.sass  import AtomicMatcher, PatternBuilder
from reprospect.utils      import cmake

from tests.python.parameters import Parameters, PARAMETERS
from tests.python.test.sass.test_sass import get_decoder

@pytest.fixture(scope = 'session')
@typeguard.typechecked
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

@pytest.fixture(scope = 'session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestAtomicMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.AtomicMatcher`.
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
    cuda::atomic_ref<{type}, cuda::thread_scope_device> ref(dst[index]);
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

    @staticmethod
    def match_one(*, decoder, **kwargs) -> tuple[AtomicMatcher, Instruction, regex.Match]:
        """
        Match exactly one instruction.
        """
        matcher = AtomicMatcher(**kwargs)
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1, matcher

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())

        return matcher, inst, matched

    @pytest.mark.parametrize(
        'word', [
            ( 16, 'short int', 'unsigned short int'),
            ( 32, 'float',     'unsigned int'),
            ( 64, 'double',    'unsigned long long int'),
    ], ids = str)
    def test_atomicCAS(self, request, workdir, word, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD_BASED_ON_CAS`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}{word[0]}.cu'
        FILE.write_text(self.CODE_ADD_BASED_ON_CAS.format(
            size = int(word[0] / 8),
            type = word[1],
            integer = word[2],
        ))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the atomic CAS.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'CAS', consistency = 'STRONG', scope = 'DEVICE',
            dtype = (None, word[0]),
        )

        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 5
        assert matched.captures('operands')[0] == 'PT'

    def test_atomicCAS_128(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
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

        kwargs = {'cwd' : workdir, 'arch' : parameters.arch, 'file' : FILE, 'cmake_file_api' : cmake_file_api}

        if expecting_failure:
            with pytest.raises(subprocess.CalledProcessError):
                get_decoder(**kwargs)
            return
        decoder, _ = get_decoder(**kwargs)

        # Find the atomic CAS.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'CAS', consistency = 'STRONG', scope = 'DEVICE',
            dtype = (None, 128),
        )

        assert {'CAS', '128'}.issubset(matched.captures('modifiers'))

    def test_add_relaxed_block_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        As of CUDA 13.0.0, the generated code still applies the ``.STRONG`` modifier,
        regardless of the ``.relaxed`` qualifier shown in the PTX.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type = 'int'))

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic add.
        matcher, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'ADD',
            consistency = 'STRONG', scope = 'BLOCK',
            memory = '',
            dtype = ('S', 32),
        )

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        if matcher.version in semantic_version.SimpleSpec('<12.8'):
            assert 'atom.add.relaxed.cta.u32' in result
        else:
            assert 'atom.add.relaxed.cta.s32' in result

    def test_add_relaxed_block_unsigned_long_long_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Similar to :py:meth:`test_add_relaxed_block_int` for `unsigned long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type = 'unsigned long long int'))

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic add.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'ADD',
            consistency = 'STRONG', scope = 'BLOCK',
            memory = '',
            dtype = ('U', 64),
        )

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.add.relaxed.cta.u64' in result, result

    def test_add_relaxed_block_float(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Similar to :py:meth:`test_add_relaxed_block_int` for `float`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type = 'float'))

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic add.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'ADD',
            consistency = 'STRONG', scope = 'BLOCK',
            memory = '',
            dtype = ('F', 32),
        )

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.add.relaxed.cta.f32' in result

    def test_add_relaxed_block_double(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Similar to :py:meth:`test_add_relaxed_block_int` for `double`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type = 'double'))

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic add.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'ADD',
            consistency = 'STRONG', scope = 'BLOCK',
            memory = '',
            dtype = ('F', 64),
        )

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.add.relaxed.cta.f64' in result

    def test_min_relaxed_device_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MIN` for `int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MIN.format(type = 'int'))

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic min.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'MIN',
            consistency = 'STRONG', scope = 'DEVICE',
            memory = '',
            dtype = ('S', 32),
        )

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'
        assert {'MIN', 'S32'}.issubset(matched.captures('modifiers'))

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.min.relaxed.gpu.s32' in result

    def test_min_relaxed_device_long_long_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MIN` for `long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MIN.format(type = 'long long int'))

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic min.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'MIN',
            consistency = 'STRONG', scope = 'DEVICE',
            memory = '',
            dtype = ('S', 64),
        )

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'
        assert {'MIN', 'S64'}.issubset(matched.captures('modifiers'))

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.min.relaxed.gpu.s64' in result

    def test_min_relaxed_device_unsigned_long_long_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MIN` for `unsigned long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MIN.format(type = 'unsigned long long int'))

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic min.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch,
            operation = 'MIN',
            consistency = 'STRONG', scope = 'DEVICE',
            memory = '',
            dtype = ('U', 64),
        )

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'
        assert {'MIN', '64'}.issubset(matched.captures('modifiers'))

        # In the PTX, we can see the '.relaxed'.
        result = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.min.relaxed.gpu.u64' in result

    def test_exch_strong_device_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_EXCH` for `int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_EXCH.format(type = 'int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the atomic exchange.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch, operation = 'EXCH', dtype = ('S', 32), scope = 'DEVICE', consistency = 'STRONG',
        )

        assert {'EXCH'}.issubset(matched.captures('modifiers'))

    def test_exch_strong_device_unsigned_long_long_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_EXCH` for `unsigned long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_EXCH.format(type = 'unsigned long long int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the atomic exchange.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch, operation = 'EXCH', dtype = ('S', 64), scope = 'DEVICE', consistency = 'STRONG',
        )

        assert {'EXCH', '64'}.issubset(matched.captures('modifiers'))

    def test_exch_strong_device_float(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_EXCH` for `float`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_EXCH.format(type = 'float'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the atomic exchange.
        _, _, matched = self.match_one(
            decoder = decoder,
            arch = parameters.arch, operation = 'EXCH', dtype = ('F', 32), scope = 'DEVICE', consistency = 'STRONG',
        )

        assert {'EXCH'}.issubset(matched.captures('modifiers'))
