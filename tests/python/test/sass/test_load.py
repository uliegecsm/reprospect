import logging
import os
import pathlib
import re
import typing

import pytest
import semantic_version

from reprospect.test.sass.instruction import (
    InstructionMatch,
    LoadConstantMatcher, LoadGlobalMatcher, LoadMatcher,
    PatternBuilder,
)
from reprospect.tools.architecture    import NVIDIAArch
from reprospect.utils                 import cmake

from tests.python.parameters                 import Parameters, PARAMETERS
from tests.python.test.sass.test_instruction import (
    get_decoder,
    CODE_ELEMENTWISE_ADD_RESTRICT,
    CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE,
    CODE_ELEMENTWISE_ADD_RESTRICT_256_WIDE,
)

class TestLoadConstantMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.LoadConstantMatcher`.
    """
    INSTRUCTIONS : typing.Final[dict[str, tuple[LoadConstantMatcher, InstructionMatch]]] = {
        'LDC R1, c[0x0][0x37c]' : (
            LoadConstantMatcher(),
            InstructionMatch(opcode = 'LDC', modifiers = (), operands = ('R1', 'c[0x0][0x37c]'), additional = {'bank' : ['0x0'], 'offset' : ['0x37c']}),
        ),
        'LDCU UR4, c[0x2][0x364]' : (
            LoadConstantMatcher(uniform = True),
            InstructionMatch(opcode = 'LDCU', modifiers = (), operands = ('UR4', 'c[0x2][0x364]'), additional = {'bank' : ['0x2'], 'offset' : ['0x364']}),
        ),
        'LDC.64 R6, c[0x1][0x398]' : (
            LoadConstantMatcher(size = 64),
            InstructionMatch(opcode = 'LDC', modifiers = ('64',), operands = ('R6', 'c[0x1][0x398]'), additional = {'bank' : ['0x1'], 'offset' : ['0x398']}),
        ),
        'LDCU.64 UR6, c[0x3][0x358]' : (
            LoadConstantMatcher(uniform = True, size = 64),
            InstructionMatch(opcode = 'LDCU', modifiers = ('64',), operands = ('UR6', 'c[0x3][0x358]'), additional = {'bank' : ['0x3'], 'offset' : ['0x358']}),
        ),
        'LDCU UR4, c[0x3][UR0]' : (
            LoadConstantMatcher(uniform = True),
            InstructionMatch(opcode = 'LDCU', modifiers = (), operands = ('UR4', 'c[0x3][UR0]'), additional = {'bank' : ['0x3'], 'offset' : ['UR0']}),
        ),
    }
    """
    Zoo of real SASS instructions.
    """

    CODE_CONSTANT_ARRAY : typing.Final[str] = """\
__constant__ {type} data[128];
__global__ __launch_bounds__(128, 1) void ldc({type}* __restrict__ const out)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index] = data[index];
}}
"""

    @pytest.mark.parametrize('instruction,matcher,expected', ((instr, *vals) for instr, vals in INSTRUCTIONS.items()))
    def test(self, instruction : str, matcher : LoadConstantMatcher, expected : InstructionMatch) -> None:
        assert (matched := matcher.match(inst = instruction)) is not None
        assert matched == expected

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_array_of_64bit_elements(self, request, workdir : pathlib.Path, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Loads of size 64 with :py:attr:`CODE_CONSTANT_ARRAY`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_CONSTANT_ARRAY.format(type = 'long'))

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the constant load.
        matcher = LoadConstantMatcher(size = 64, uniform = False)
        load_cst = tuple((inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst)))
        assert len(load_cst) == 1, matcher

        inst, matched = load_cst[0]

        logging.info(f'{matcher} matched instruction {inst.instruction} as {matched}.')

        assert len(matched.operands) == 2
        assert re.match(PatternBuilder.CONSTANT, matched.operands[1]) is not None

        assert matched.additional is not None
        assert 'bank' in matched.additional
        assert 'offset' in matched.additional
        assert re.match(PatternBuilder.REG, matched.additional['offset'][0]) is not None

class TestLoadMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.LoadMatcher`
    and :py:class:`reprospect.test.sass.instruction.LoadGlobalMatcher`.
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
    def test(self) -> None:
        matcher = LoadMatcher(arch = NVIDIAArch.from_compute_capability(100), size = 64, memory = None)
        assert matcher.match(inst = 'LD.E.64 R2, desc[UR10][R4.64]') is not None

        matcher = LoadMatcher(arch = NVIDIAArch.from_compute_capability(120), size = 256, readonly = True, memory = 'G')
        assert matcher.match(inst = 'LDG.E.ENL2.256.CONSTANT R12, R8, desc[UR4][R2.64]') is not None

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_elementwise_add_restrict(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test loads with :py:const:`tests.python.test.sass.test_instruction.CODE_ELEMENTWISE_ADD_RESTRICT`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT)

        decoder, _ = get_decoder(cwd = workdir,arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the read-only load.
        matcher = LoadGlobalMatcher(arch = parameters.arch, readonly = True)
        load_ro = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(load_ro) == 1

        inst_ro, matched_ro = load_ro[0]

        logging.info(f'{matcher} matched instruction {inst_ro.instruction} as {matched_ro}.')

        assert 'CONSTANT' in matched_ro.modifiers
        assert len(matched_ro.additional['address']) == 1
        assert len(matched_ro.operands) == 2

        # Find the load.
        matcher = LoadGlobalMatcher(arch = parameters.arch, readonly = False)
        load = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(load) == 1

        inst, matched = load[0]

        logging.info(f'{matcher} matched instruction {inst.instruction} as {matched}.')

        assert 'CONSTANT' not in matched.modifiers
        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

        assert inst_ro != inst

        # Find both loads by letting the 'CONSTANT' be optional.
        matcher = LoadGlobalMatcher(arch = parameters.arch, readonly = None)
        load = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(load) == 2

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_elementwise_add_restrict_128_wide(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test 128-bits wide loads with
        :py:const:`tests.python.test.sass.test_instruction.CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE)

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

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_elementwise_add_restrict_256_wide(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test 256-bits wide loads with
        :py:const:`tests.python.test.sass.test_instruction.CODE_ELEMENTWISE_ADD_RESTRICT_256_WIDE`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_256_WIDE)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Before CUDA 13, double4 is always 16-bytes aligned.
        aligned_16 : typing.Final[bool] = semantic_version.Version(os.environ['CUDA_VERSION']) in semantic_version.SimpleSpec('<13')

        # Find the read-only wide load(s).
        # Before BLACKWELL, there must be two 128-bits loads.
        matcher_lro_128 = LoadGlobalMatcher(arch = parameters.arch, size = 128, readonly = True)
        matcher_lro_256 = LoadGlobalMatcher(arch = parameters.arch, size = 256, readonly = True)
        lro_128 = tuple(filter(matcher_lro_128, decoder.instructions))
        lro_256 = tuple(filter(matcher_lro_256, decoder.instructions))
        if parameters.arch.compute_capability < 100 or aligned_16:
            assert len(lro_128) == 2 and len(lro_256) == 0
        else:
            assert len(lro_128) == 0 and len(lro_256) == 1

        # Find the wide load(s).
        # Before BLACKWELL, there must be two 128-bits loads.
        matcher_l_128 = LoadGlobalMatcher(arch = parameters.arch, size = 128, readonly = False)
        matcher_l_256 = LoadGlobalMatcher(arch = parameters.arch, size = 256, readonly = False)
        l_128 = tuple(filter(matcher_l_128, decoder.instructions))
        l_256 = tuple(filter(matcher_l_256, decoder.instructions))
        if parameters.arch.compute_capability < 100 or aligned_16:
            assert len(l_128) == 2 and len(l_256) == 0
        else:
            assert len(l_128) == 0 and len(l_256) == 1

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
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
