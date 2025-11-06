import functools
import logging
import os
import pathlib
import typing

import pytest
import regex
import semantic_version
import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.sass         import Decoder
from reprospect.utils              import cmake
from reprospect.test.sass          import AtomicMatcher, \
                                          FloatAddMatcher, \
                                          OpcodeModsMatcher, \
                                          OpcodeModsWithOperandsMatcher, \
                                          LoadGlobalMatcher, \
                                          PatternBuilder, \
                                          ReductionMatcher, \
                                          StoreGlobalMatcher

from tests.python.parameters          import Parameters, PARAMETERS
from tests.python.tools.test_binaries import get_compilation_output

@pytest.fixture(scope = 'session')
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

@pytest.fixture(scope = 'session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

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
def get_decoder(*, cwd : pathlib.Path, arch : NVIDIAArch, file : pathlib.Path, cmake_file_api : cmake.FileAPI, **kwargs) -> tuple[Decoder, pathlib.Path]:
    """
    Compile the code in `file` for `arch` and return a :py:class:`reprospect.tools.sass.Decoder`.
    """
    output, _ = get_compilation_output(
        source = file,
        cwd = cwd,
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
    def test_elementwise_add_restrict(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test loads with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT)

        decoder, _ = get_decoder(cwd = workdir,arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

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
    def test_elementwise_add_restrict_wide(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test wide loads with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT_WIDE`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_WIDE)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

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

    def test_constant(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
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
            FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.{name}.cu'
            FILE.write_text(code)

            decoders[name], _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        assert len(list(filter(LoadGlobalMatcher(arch = parameters.arch, readonly = True), decoders['restrict'   ].instructions))) == 1
        assert len(list(filter(LoadGlobalMatcher(arch = parameters.arch, readonly = True), decoders['no_restrict'].instructions))) == 0
        assert len(list(filter(LoadGlobalMatcher(arch = parameters.arch, readonly = True), decoders['ldg'        ].instructions))) == 1

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestStoreGlobalMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.StoreGlobalMatcher`.
    """
    @typeguard.typechecked
    def test_elementwise_add_restrict(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test store with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

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

    def test_elementwise_add_restrict_wide(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test wide store with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT_WIDE`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_WIDE)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

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
    def test_elementwise_add_restrict_wide(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:const:`CODE_ELEMENTWISE_ADD_RESTRICT_WIDE`.

        There will be 4 ``FADD`` instructions because of the *float4*.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_WIDE)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

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

    def test_add_strong_device_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'ADD', scope = 'DEVICE', consistency = 'STRONG', dtype = ('S', 32))
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert {'ADD'}.issubset(matched.captures('modifiers'))
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

        # Another consistency would fail.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'ADD', scope = 'DEVICE', consistency = 'WEAK', dtype = ('S', 32))
        assert not any(matcher.matches(inst) for inst in decoder.instructions)

    def test_add_strong_device_unsigned_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `unsigned int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'unsigned int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'ADD', scope = 'DEVICE', consistency = 'STRONG', dtype = ('U', 32))
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert {'ADD'}.issubset(matched.captures('modifiers'))
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

    def test_add_strong_device_unsigned_long_long_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `unsigned long long int`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'unsigned long long int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(
            arch = parameters.arch,
            operation = 'ADD',
            scope = 'DEVICE',
            consistency = 'STRONG',
            dtype = ('U', 64),
        )
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert len(matched.captures('opcode')) == 1
        assert 'modifiers' in matched.capturesdict()
        assert {'ADD', '64'}.issubset(matched.captures('modifiers'))
        assert len(matched.captures('address')) == 1
        assert len(matched.captures('operands')) == 2

    def test_add_strong_device_float(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `float`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'float'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

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

    def test_add_strong_device_double(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_ADD` for `double`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD.format(type = 'double'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

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

    def test_sub_strong_device(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_SUB`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_SUB)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        # Note that the source is negated in another instruction.
        red = list(filter(ReductionMatcher(arch = parameters.arch, operation = 'ADD', scope = 'DEVICE', consistency = 'STRONG'), decoder.instructions))
        assert len(red) == 1

        logging.info(red[0])

    def test_max_strong_device_long_long_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `long long int`. The modifier is ``MAX.S64``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type = 'long long int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher_type : typing.Type[AtomicMatcher] | typing.Type[ReductionMatcher]
        if semantic_version.Version(os.environ['CUDA_VERSION']) in semantic_version.SimpleSpec('<12.8'):
            matcher_type = AtomicMatcher
        else:
            matcher_type = ReductionMatcher
        matcher = matcher_type(arch = parameters.arch, operation = 'MAX', dtype = ('S', 64), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert {'MAX', 'S64'}.issubset(matched.captures('modifiers'))

    def test_max_strong_device_unsigned_long_long_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `unsigned long long int`. The modifier is ``MAX.64``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type = 'unsigned long long int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher_type : typing.Type[AtomicMatcher] | typing.Type[ReductionMatcher]
        if semantic_version.Version(os.environ['CUDA_VERSION']) in semantic_version.SimpleSpec('<12.8'):
            matcher_type = AtomicMatcher
        else:
            matcher_type = ReductionMatcher
        matcher = matcher_type(arch = parameters.arch, operation = 'MAX', dtype = ('U', 64), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert {'MAX', '64'}.issubset(matched.captures('modifiers'))

    def test_max_strong_device_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `int`. The modifier is ``MAX.S32``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type = 'int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'MAX', dtype = ('S', 32), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert {'MAX', 'S32'}.issubset(matched.captures('modifiers'))

    def test_max_strong_device_unsigned_int(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_MAX` for `unsigned int`. The modifier is ``MAX``.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_MAX.format(type = 'unsigned int'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the reduction.
        matcher = ReductionMatcher(arch = parameters.arch, operation = 'MAX', dtype = ('U', 32), scope = 'DEVICE', consistency = 'STRONG')
        red = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        assert len(red) == 1

        inst, matched = red[0]

        logging.info(matcher.pattern)
        logging.info(inst.instruction)
        logging.info(matched.capturesdict())
        assert {'MAX'}.issubset(matched.captures('modifiers'))

class TestOpcodeModsMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.OpcodeModsMatcher`.
    """
    def test_with_square_brackets(self):
        instruction = 'IMAD R4, R4, c[0x0][0x0], R3'

        matched = OpcodeModsMatcher(instruction = 'IMAD').matches(instruction)

        assert matched is not None
        assert matched.captures('operands') == ['R4', 'R4', 'c[0x0][0x0]', 'R3']

    def test_with_minus_sign(self):
        instruction = 'UIADD3 UR5, UPT, UPT, -UR4, UR9, URZ'

        matched = OpcodeModsMatcher(instruction = 'UIADD3').matches(instruction)

        assert matched is not None
        assert matched.captures('operands') == ['UR5', 'UPT', 'UPT', '-UR4', 'UR9', 'URZ']

    def test_with_reuse(self):
        instruction = 'ISETP.GE.U32.AND P0, PT, R0.reuse, UR5, PT'

        matched = OpcodeModsMatcher(instruction = 'ISETP.GE.U32.AND').matches(instruction)

        assert matched is not None
        assert matched.captures('operands') == ['P0', 'PT', 'R0.reuse', 'UR5', 'PT']

    def test_with_descr(self):
        instruction = 'LDG.E R2, desc[UR6][R2.64]'

        matched = OpcodeModsMatcher(instruction = 'LDG.E').matches(instruction)

        assert matched is not None
        assert matched.captures('operands') == ['R2', 'desc[UR6][R2.64]']

class TestOpcodeModsWithOperandsMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.OpcodeModsWithOperandsMatcher`.
    """
    def test(self):
        instruction = 'ISETP.NE.AND P2, PT, R4, RZ, PT'

        matcher = OpcodeModsWithOperandsMatcher(instruction = 'ISETP.NE.AND', operands = (
            PatternBuilder.PRED,
            PatternBuilder.PREDT,
            'R4',
            PatternBuilder.REGZ,
            PatternBuilder.PREDT,
        ))

        logging.info(matcher.pattern)

        matched = matcher.matches(instruction)

        assert matched is not None
        assert matched.captures('operands') == ['P2', 'PT', 'R4', 'RZ', 'PT']
