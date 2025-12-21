import logging
import pathlib
import re
import typing

import pytest

from reprospect.test.sass.composite import instructions_contain
from reprospect.test.sass.instruction import OpcodeModsWithOperandsMatcher
from reprospect.test.sass.instruction.immediate import Immediate
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass import Decoder
from reprospect.utils import cmake

from tests.compilation import get_compilation_output
from tests.parameters import PARAMETERS, Parameters


@pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
class TestImmediateFromCompileOutput:
    CODE_ADD_FLOATING_LITERAL: typing.Final[str] = """\
__global__ void add_immediate_{suffix}_kernel(float* const a, const unsigned int size)
{{
    const float cst_{suffix} = {value};
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        a[index] += cst_{suffix};
}}
"""

    FLOATING_LITERALS: typing.Final[list[str]] = [
        '3.14f', # 3.1400001049041748047 (decimal notation)
        '1e10f', # 1.00000000000000000000e+10 (scientific notation)
        '.5f',   # 0.5 (zero integer part)
        '1.e2f', # 100 (zero fractional part)
        '-.5f',  # -0.5 (negative, zero integer part)
        '__int_as_float(0x7fc00000)',
        #'CUDART_NAN_F', # 0x7fffffff
        'CUDART_INF_F', # +INF
        '-CUDART_NAN_F', # -QNAN
        '-CUDART_INF_F', # -INF
    ]
    MATCHER: typing.Final[OpcodeModsWithOperandsMatcher] = OpcodeModsWithOperandsMatcher(
        opcode='FADD', operands=(PatternBuilder.REG, PatternBuilder.REG, Immediate.FLOATING_OR_LIMIT),
    )

    def test(self, request, workdir: pathlib.Path, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'

        FILE.write_text(
            '#include "math_constants.h"\n' +
            '\n'.join(self.CODE_ADD_FLOATING_LITERAL.format(value=value, suffix=str(idx))
                      for idx, value in enumerate(self.FLOATING_LITERALS)
        ))

        output, _ = get_compilation_output(
            source=FILE,
            cwd=workdir,
            arch=parameters.arch,
            object_file=True,
            resource_usage=False,
            cmake_file_api=cmake_file_api,
        )

        cuobjdump = CuObjDump(file=output, arch=parameters.arch, sass=True)
        assert len(cuobjdump.functions) == len(self.FLOATING_LITERALS)

        for idx in range(len(self.FLOATING_LITERALS)):
            [sig] = [sig for sig in cuobjdump.functions if re.search(f'add_immediate_{idx}_kernel', sig) is not None]
            decoder = Decoder(code=cuobjdump.functions[sig].code)
            instructions = decoder.instructions

            matched = instructions_contain(matcher=self.MATCHER).assert_matches(instructions)
            logging.info(f'Instruction match for add_immediate_{idx}_kernel:\n{matched}')

class TestImmediate:
    """
    Check immediate patterns.
    """

    IMMEDIATES: typing.Final[tuple[str,...]] = (
        '3.1400001049041748047', # '3.14f'
        '1.00000000000000000000e+10', # '1e10f'
        '0.5', # '.5f'
        '100', # '1.e2f'
        '-0.5', # '-.5f'
        '+QNAN', # '__int_as_float(0x7fc00000)',
        '0x7fffffff', # 'CUDART_NAN_F'
        '+INF', # 'CUDART_INF_F'
        '-QNAN', # '-CUDART_NAN_F'
        '-INF', # '-CUDART_INF_F'
    )
        
    def test(self) -> None:
        assert re.fullmatch(Immediate.INF, '-INF') is not None
        assert re.fullmatch(Immediate.INF, '100') is None

        assert re.fullmatch(Immediate.QNAN, '+QNAN') is not None
        assert re.fullmatch(Immediate.QNAN, '0x7fffffff') is not None
        assert re.fullmatch(Immediate.QNAN, '100') is None

        assert re.fullmatch(Immediate.FLOATING, '3.14') is not None
        assert re.fullmatch(Immediate.FLOATING, '+QNAN') is None

        assert all(re.fullmatch(Immediate.FLOATING_OR_LIMIT, immediate) is not None for immediate in self.IMMEDIATES)
        assert re.fullmatch(Immediate.FLOATING_OR_LIMIT, '0x3') is None
