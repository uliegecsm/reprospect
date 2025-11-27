import logging
import pathlib
import re
import typing

import pytest

from reprospect.test.sass.instruction import InstructionMatch, LoadConstantMatcher, PatternBuilder
from reprospect.utils                 import cmake

from tests.python.parameters                 import Parameters, PARAMETERS
from tests.python.test.sass.test_instruction import get_decoder

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
        assert (matched := matcher.matches(inst = instruction)) is not None
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
        load_cst = tuple((inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst)))
        assert len(load_cst) == 1, matcher

        inst, matched = load_cst[0]

        logging.info(f'{matcher} matched instruction {inst.instruction} as {matched}.')

        assert len(matched.operands) == 2
        assert re.match(PatternBuilder.CONSTANT, matched.operands[1]) is not None

        assert matched.additional is not None
        assert 'bank' in matched.additional
        assert 'offset' in matched.additional
        assert re.match(PatternBuilder.REG, matched.additional['offset'][0]) is not None
