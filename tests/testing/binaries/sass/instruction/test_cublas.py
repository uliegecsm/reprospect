import dataclasses
import logging
import pathlib
import re
import typing

import pytest
import regex

from reprospect.testing.binaries.sass.instruction import (
    AddressMatcher,
    AnyMatcher,
    ConstantMatcher,
    Immediate,
    MemorySpace,
    RegisterMatcher,
)
from reprospect.tools.binaries.sass.decoder import Decoder, Instruction
from reprospect.utils import cmake

from tests.cublas import CuBLAS
from tests.parameters import PARAMETERS, Parameters

FIXIT_FILTER: typing.Final[re.Pattern[str]] = re.compile(r'\] \[| ,')
FIXIT_REPLACEMENTS: typing.Final[dict[str, str]] = {
    '] [': '][',
    ' ,': ',',
}

def fixit(instruction: Instruction) -> Instruction:
    """
    Fix the few instructions that do not conform to the expected format.

        +------------+--------+
        | Before     | After  |
        +============+========+
        | ``]␣[``    | ``][`` |
        +------------+--------+
        | :code:`␣,` | ``,``  |
        +------------+--------+

    Here are a few examples:

    * :code:`HADD2.F32 R10, -RZ, c[0x0]␣[0x160].H0_H0` becomes :code:`HADD2.F32 R10, -RZ, c[0x0][0x160].H0_H0`.
    * :code:`FSETP.GEU.AND P1, PT, |R151|, +INF␣, PT` becomes :code:`FSETP.GEU.AND P1, PT, |R151|, +INF, PT`.

    These cases are rare (a handful out of millions of instructions), so the
    search acts as a fast-path guard before the substitution.
    """
    if FIXIT_FILTER.search(instruction.instruction) is not None:
        output = FIXIT_FILTER.sub(
            repl=lambda m: FIXIT_REPLACEMENTS[m.group()],
            string=instruction.instruction,
        )
        return dataclasses.replace(instruction, instruction=output)
    return instruction

class TestCuBLAS:
    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test_operands(self, workdir: pathlib.Path, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
        """
        For each supported architecture, pick one cubin from cuBLAS.
        Then, loop over all instructions of each function, and match each operand.
        """
        cublas = CuBLAS(cmake_file_api=cmake_file_api)
        try:
            cuobjdump = cublas.cuobjdump(arch=parameters.arch, cwd=workdir, sass=True)
        except IndexError:
            pytest.skip(f'The library {cublas.libcublas} does not contain any CUDA binary file for {parameters.arch}.')

        if len(cuobjdump.functions) == 0:
            pytest.skip(f'{cuobjdump.file} does not contain any function for {parameters.arch}.')

        # Match any instruction.
        matcher = AnyMatcher()

        matchers_op: list = []

        # Matchers for operands that are well covered.
        matchers_op.extend((
            regex.compile(AddressMatcher.build_pattern(arch=parameters.arch, memory=MemorySpace.GENERIC, captured=False)),
            regex.compile(AddressMatcher.build_pattern(arch=parameters.arch, memory=MemorySpace.GLOBAL, captured=False)),
            regex.compile(AddressMatcher.build_pattern(arch=parameters.arch, memory=MemorySpace.LOCAL, captured=False)),
            regex.compile(AddressMatcher.build_pattern(arch=parameters.arch, memory=MemorySpace.SHARED, captured=False)),
            regex.compile(ConstantMatcher.build_pattern(captured=False)),
            regex.compile(Immediate.FLOATING_OR_LIMIT),
            regex.compile(RegisterMatcher.build_pattern(captured=False)),
        ))

        # Matchers for other operands.
        matchers_op.extend((
            regex.compile(r'B[0-9]+'),
            regex.compile(r'SB[0-9]'),
            regex.compile(r'PR'),
            regex.compile(r'(SR_TID|SR_CTAID)\.(X|Y|Z)'),
            regex.compile(r'SR(X|Y|Z)'),
            regex.compile(r'SR_CgaCtaId|SR_LANEID|SR_LTMASK|SR_SWINLO|SR_SWINHI'),
        ))

        for name, function in cuobjdump.functions.items():
            logging.info(f'Picked function {name} from {cuobjdump.file} for {parameters.arch}.')

            decoder = Decoder(code=function.code)

            for instruction in decoder.instructions:
                instruction = fixit(instruction=instruction)
                assert (matched := matcher.match(instruction)) is not None
                for operand in matched.operands:
                    if not any(mop.match(operand) is not None for mop in matchers_op):
                        logging.info(f'Instruction {instruction.instruction} has unmatched operand {operand}.')
                        pytest.fail()
