import functools
import logging
import os
import pathlib
import subprocess

import pytest
import regex
import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.sass         import Decoder
from reprospect.utils              import cmake
from reprospect.test.sass          import AtomicMatcher, FloatAddMatcher, LoadGlobalMatcher, PatternBuilder, ReductionMatcher, StoreGlobalMatcher

from tests.python.tools.test_binaries import Parameters, PARAMETERS, get_compilation_output

@pytest.fixture(scope = 'session')
@typeguard.typechecked
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

CODE_ELEMENTWISE_ADD_RESTRICT = """\
__global__ void elementwise_add_restrict(int* __restrict__ const dst, const int* __restrict__ const src) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] += src[index];
}
"""
"""Element-wise add with 32-bit ``int``."""

CODE_ELEMENTWISE_ADD_RESTRICT_WIDE = """\
__global__ void elementwise_add_restrict_wide(float4* __restrict__ const dst, const float4* __restrict__ const src) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const float4& a = src[index];
    const float4& b = dst[index];
    dst[index] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
"""
"""Element-wise add with 128-bit ``float4``."""

@functools.lru_cache(maxsize = 128)
@typeguard.typechecked
def get_decoder(*, arch : NVIDIAArch, file : pathlib.Path, cmake_file_api : cmake.FileAPI, **kwargs) -> tuple[Decoder, pathlib.Path]:
    """
    Compile the code in `file` for `arch` and return a :py:class:`reprospect.tools.sass.Decoder`.
    """
    output, _ = get_compilation_output(
        source = file,
        cwd = TMPDIR,
        arch = arch,
        object = True,
        resource_usage = False,
        cmake_file_api = cmake_file_api,
        **kwargs,
    )

    return Decoder(code = CuObjDump(file = output, arch = arch, sass = True).sass), output

class TestPatternBuilder:
    """
    Tests for :py:class:`reprospect.test.sass.PatternBuilder`.
    """
    def test_reg_addr(self):
        """
        Test :py:meth`reprospect.test.sass.PatternBuilder.reg_addr`.
        """
        # Defaulted offset behaviour.
        pattern = PatternBuilder.reg_addr()

        matched = regex.match(pattern, string = '[R0]')
        assert matched is not None
        assert matched.group('address')  == 'R0' and matched.captures('address')  == ['R0']
        assert matched.group('operands') == 'R0' and matched.captures('operands') == ['R0']

        matched = regex.match(pattern, string = '[R2+0x10]')
        assert matched is not None
        assert matched.group('address')  == 'R2+0x10' and matched.captures('address')  == ['R2+0x10']
        assert matched.group('operands') == 'R2+0x10' and matched.captures('operands') == ['R2+0x10']

        # Enforced offset behaviour.
        pattern = PatternBuilder.reg_addr(True)

        assert regex.match(pattern, string = '[R0]') is None

        matched = regex.match(pattern, string = '[R2+0x10]')
        assert matched is not None
        assert matched.group('address')  == 'R2+0x10' and matched.captures('address')  == ['R2+0x10']
        assert matched.group('operands') == 'R2+0x10' and matched.captures('operands') == ['R2+0x10']

        # Disabled offset behaviour.
        pattern = PatternBuilder.reg_addr(False)

        matched = regex.match(pattern, string = '[R0]')
        assert matched is not None
        assert matched.group('address')  == 'R0' and matched.captures('address')  == ['R0']
        assert matched.group('operands') == 'R0' and matched.captures('operands') == ['R0']

        assert regex.match(pattern, string = '[R2+0x10]') is None

    def test_reg64_addr(self):
        """
        Test :py:meth`reprospect.test.sass.PatternBuilder.reg64_addr`.
        """
        # Default offset behaviour.
        pattern = PatternBuilder.reg64_addr()

        matched = regex.match(pattern, string = '[R0.64]')
        assert matched is not None
        assert matched.group('address')  == 'R0.64' and matched.captures('address')  == ['R0.64']
        assert matched.group('operands') == 'R0.64' and matched.captures('operands') == ['R0.64']

        matched = regex.match(pattern, string = '[R2.64+0x10]')
        assert matched is not None
        assert matched.group('address')  == 'R2.64+0x10' and matched.captures('address')  == ['R2.64+0x10']
        assert matched.group('operands') == 'R2.64+0x10' and matched.captures('operands') == ['R2.64+0x10']

        # Enforced offset behaviour.
        pattern = PatternBuilder.reg64_addr(True)

        assert regex.match(pattern, string = '[R0.64]') is None

        matched = regex.match(pattern, string = '[R2.64+0x10]')
        assert matched is not None
        assert matched.group('address')  == 'R2.64+0x10' and matched.captures('address')  == ['R2.64+0x10']
        assert matched.group('operands') == 'R2.64+0x10' and matched.captures('operands') == ['R2.64+0x10']

        # Disabled offset behaviour.
        pattern = PatternBuilder.reg64_addr(False)

        matched = regex.match(pattern, string = '[R0.64]')
        assert matched is not None
        assert matched.group('address')  == 'R0.64' and matched.captures('address')  == ['R0.64']
        assert matched.group('operands') == 'R0.64' and matched.captures('operands') == ['R0.64']

        assert regex.match(pattern, string = '[R2.64+0x10]') is None

    def test_desc_reg64_addr(self):
        """
        Test :py:meth`reprospect.test.sass.PatternBuilder.desc_reg64_addr`.
        """
        # Default offset behaviour.
        pattern = PatternBuilder.desc_reg64_addr()

        matched = regex.match(pattern, string = 'desc[UR86][R0.64]')
        assert matched is not None
        assert matched.group('address')  == 'desc[UR86][R0.64]' and matched.captures('address')  == ['desc[UR86][R0.64]']
        assert matched.group('operands') == 'desc[UR86][R0.64]' and matched.captures('operands') == ['desc[UR86][R0.64]']

        matched = regex.match(pattern, string = 'desc[UR86][R2.64+0x10]')
        assert matched is not None
        assert matched.group('address')  == 'desc[UR86][R2.64+0x10]' and matched.captures('address')  == ['desc[UR86][R2.64+0x10]']
        assert matched.group('operands') == 'desc[UR86][R2.64+0x10]' and matched.captures('operands') == ['desc[UR86][R2.64+0x10]']

        # Enforced offset behaviour.
        pattern = PatternBuilder.desc_reg64_addr(True)

        assert regex.match(pattern, string = 'desc[UR86][R0.64]') is None

        matched = regex.match(pattern, string = 'desc[UR86][R2.64+0x10]')
        assert matched is not None
        assert matched.group('address')  == 'desc[UR86][R2.64+0x10]' and matched.captures('address')  == ['desc[UR86][R2.64+0x10]']
        assert matched.group('operands') == 'desc[UR86][R2.64+0x10]' and matched.captures('operands') == ['desc[UR86][R2.64+0x10]']

        # Disabled offset behaviour.
        pattern = PatternBuilder.desc_reg64_addr(False)

        matched = regex.match(pattern, string = 'desc[UR86][R0.64]')
        assert matched is not None
        assert matched.group('address')  == 'desc[UR86][R0.64]' and matched.captures('address')  == ['desc[UR86][R0.64]']
        assert matched.group('operands') == 'desc[UR86][R0.64]' and matched.captures('operands') == ['desc[UR86][R0.64]']

        assert regex.match(pattern, string = 'desc[UR86][R2.64+0x10]') is None

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
class TestLoadGlobalMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.LoadGlobalMatcher`.
    """
    CODE_ELEMENTWISE_ADD = """\
__global__ void elementwise_add(int* const dst, const int* const src) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] += src[index];
}
"""

    CODE_ELEMENTWISE_ADD_LDG = """\
__global__ void elementwise_add_ldg(int* const dst, const int* const src) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] += __ldg(&src[index]);
}
"""

    @typeguard.typechecked
    def test_elementwise_add_restrict(self, request,parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test loads with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT)

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the read-only load.
        matcher = LoadGlobalMatcher(arch = parameters.arch, readonly = True)
        load_ro = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(load_ro) == 1

        inst_ro, matched_ro = load_ro[0]

        logging.info(matcher.pattern)
        logging.info(inst_ro.instruction)
        logging.info(matched_ro.capturesdict())
        assert len(matched_ro.captures('opcode')) == 1
        assert 'CONSTANT' in matched_ro.captures('modifiers')
        assert len(matched_ro.captures('address')) == 1
        assert len(matched_ro.captures('operands')) == 2

        # Find the load.
        matcher = LoadGlobalMatcher(arch = parameters.arch, readonly = False)
        load = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(load) == 1

        inst, matched = load[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'CONSTANT' not in matched.captures('modifiers')
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

        assert inst_ro != inst

        # Find both loads by letting the 'CONSTANT' be optional.
        matcher = LoadGlobalMatcher(arch = parameters.arch, readonly = None)
        load = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(load) == 2

    @typeguard.typechecked
    def test_elementwise_add_restrict_wide(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test wide loads with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT_WIDE`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_WIDE)

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the read-only wide load.
        matcher = LoadGlobalMatcher(arch = parameters.arch, size = 128, readonly = True)
        load_ro = list(filter(matcher, decoder.instructions))
        assert len(load_ro) == 1, matcher

        logging.info(load_ro[0])

        # Find the wide load.
        matcher = LoadGlobalMatcher(arch = parameters.arch, size = 128, readonly = False)
        load = list(filter(matcher, decoder.instructions))
        assert len(load) == 1

        logging.info(load[0])

        assert load_ro != load

    def test_constant(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        If `src` is declared ``const __restrict__``, the compiler is able to use the ``.CONSTANT`` modifier.
        Otherwise, we need to explicitly use ``__ldg`` to end up using ``.CONSTANT``.
        """
        ITEMS = {
            'restrict'    : CODE_ELEMENTWISE_ADD_RESTRICT,
            'no_restrict' : self.CODE_ELEMENTWISE_ADD,
            'ldg'         : self.CODE_ELEMENTWISE_ADD_LDG,
        }
        decoders = {}
        for name, code in ITEMS.items():
            FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.{name}.cu'
            FILE.write_text(code)

            decoders[name], _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        assert len(list(filter(LoadGlobalMatcher(arch = parameters.arch, readonly = True), decoders['restrict'   ].instructions))) == 1
        assert len(list(filter(LoadGlobalMatcher(arch = parameters.arch, readonly = True), decoders['no_restrict'].instructions))) == 0
        assert len(list(filter(LoadGlobalMatcher(arch = parameters.arch, readonly = True), decoders['ldg'        ].instructions))) == 1

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestStoreGlobalMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.StoreGlobalMatcher`.
    """
    @typeguard.typechecked
    def test_elementwise_add_restrict(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test store with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT)

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the store.
        matcher = StoreGlobalMatcher(arch = parameters.arch)
        store = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(store) == 1

        inst, matched = store[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

    def test_elementwise_add_restrict_wide(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test wide store with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT_WIDE`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_WIDE)

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the wide store.
        matcher = StoreGlobalMatcher(arch = parameters.arch, size = 128)
        store = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(store) == 1

        inst, matched = store[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert '128' in matched.captures('modifiers')
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestFloatAddMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.FloatAddMatcher`.
    """
    def test_elementwise_add_restrict_wide(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT_WIDE`.

        There will be 4 ``FADD`` instructions because of the *float4*.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_WIDE)

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        matcher = FloatAddMatcher()
        fadd = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(fadd) == 4

        logging.info(matcher.pattern)

        for (inst, matched) in fadd:
            logging.info(inst.instruction)
            logging.info(matched.capturesdict())
            assert all(x in matched.capturesdict() for x in ['opcode', 'operands', 'dst'])
            assert inst.instruction.startswith(matched.group('opcode'))
            assert len(matched.captures('operands')) == 3
            assert all(operand in inst.instruction for operand in matched.captures('operands'))

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestReductionMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.ReductionMatcher`.
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

    def test_add_int_strong_gpu(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `int`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'int'))

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'ADD', scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

        # Another consistency would fail.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'ADD', scope = 'DEVICE', consistency = 'WEAK')
        assert not any(matcher.matches(inst) for inst in decoder.instructions)

    def test_add_float_strong_gpu(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `float`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'float'))

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'ADD', dtype = ('F', 32), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1, matcher

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert {'F32', 'FTZ', 'RN'}.issubset(matched.captures('modifiers'))
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

    def test_add_double_strong_gpu(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `double`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'double'))

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'ADD', dtype = ('F', 64), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1, matcher

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert {'F64', 'RN'}.issubset(matched.captures('modifiers'))
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

    def test_sub_strong_gpu(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_SUB`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_SUB)

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        # Note that the source is negated in another instruction.
        red = list(filter(ReductionMatcher(arch = parameters.arch, operation = 'ADD', scope = 'DEVICE', consistency = 'STRONG'), decoder.instructions))
        assert len(red) == 1

        logging.info(red[0])

    def test_max_strong_gpu(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX`.

        Depending on the type size, the modifier is:

        * `MAX.S32` (*e.g.* `int`)
        * `MAX.S64` (*e.g.* `long long int`)
        """
        # Get instructions for S64.
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.s64.cu'
        FILE.write_text(self.CODE_MAX.format(type = 'long long int'))

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'MAX', dtype = ('S', 64), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert {'MAX', 'S64'}.issubset(matched.captures('modifiers'))

        # Get instructions for S32.
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.s32.cu'
        FILE.write_text(self.CODE_MAX.format(type = 'int'))

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'MAX', dtype = ('S', 32), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert {'MAX', 'S32'}.issubset(matched.captures('modifiers'))

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

    @pytest.mark.parametrize(
        'word', [
            ( 16, 'short int', 'unsigned short int'),
            ( 32, 'float',     'unsigned int'),
            ( 64, 'double',    'unsigned long long int'),
    ], ids = str)
    def test_atomicCAS(self, request, word, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD_BASED_ON_CAS`.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}{word[0]}.cu'
        FILE.write_text(self.CODE_ADD_BASED_ON_CAS.format(
            size = int(word[0] / 8),
            type = word[1],
            integer = word[2],
        ))

        decoder, _ = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the atomic CAS.
        matcher = AtomicMatcher(
            arch = parameters.arch,
            operation = 'CAS', consistency = 'STRONG', scope = 'DEVICE',
            dtype = (None, word[0]),
        )
        cas = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(cas) == 1

        inst, matched = cas[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 4

    def test_atomicCAS_128(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Supported from compute capability 9.x.
        """
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.128.cu'
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
                raise ValueError(f'unsupported compiler {cmake_file_api.toolchains['CUDA']['compiler']}')

        kwargs = {'arch' : parameters.arch, 'file' : FILE, 'cmake_file_api' : cmake_file_api}

        if expecting_failure:
            with pytest.raises(subprocess.CalledProcessError):
                get_decoder(**kwargs)
            return
        decoder, _ = get_decoder(**kwargs)

        # Find the atomic CAS.
        matcher = AtomicMatcher(
            arch = parameters.arch,
            operation = 'CAS', consistency = 'STRONG', scope = 'DEVICE',
            dtype = (None, 128),
        )
        cas = list(filter(matcher, decoder.instructions))
        assert len(cas) == 1, matcher.pattern

        logging.info(cas[0])

    def test_add_relaxed_block(self, request, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        As of CUDA 13.0.0, the generated code still applies the ``.STRONG`` modifier,
        regardless of the ``.relaxed`` qualifier shown in the PTX.
        """
        # Get instructions for 'int'.
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.int.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type = 'int'))

        decoder, output = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic add.
        matcher = AtomicMatcher(
            arch = parameters.arch,
            operation = 'ADD',
            consistency = 'STRONG', scope = 'BLOCK',
            memory = '',
            dtype = ('S', 32),
        )
        add = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(add) == 1, matcher

        inst, matched = add[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        output = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.add.relaxed.cta.s32' in output

        # Get instructions for 'double'.
        FILE = TMPDIR / f'{request.node.originalname}.{parameters.arch.as_sm}.double.cu'
        FILE.write_text(self.CODE_ADD_RELAXED_BLOCK.format(type = 'double'))

        decoder, output = get_decoder(arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        # Find the atomic add.
        matcher = AtomicMatcher(
            arch = parameters.arch,
            operation = 'ADD',
            consistency = 'STRONG', scope = 'BLOCK',
            memory = '',
            dtype = ('F', 64),
        )
        add = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(add) == 1, matcher

        inst, matched = add[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())

        assert regex.match(PatternBuilder.PREDT, matched.captures('operands')[0]) is not None
        assert matched.captures('operands')[1] == 'RZ'

        # In the PTX, we can see the '.relaxed'.
        output = subprocess.check_output(['cuobjdump', '--dump-ptx', output]).decode()

        assert 'atom.add.relaxed.cta.f64' in output
