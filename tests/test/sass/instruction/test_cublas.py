import logging
import pathlib
import random
import typing

import pytest
import regex

from reprospect.test.sass.instruction.address import AddressMatcher, MemorySpace
from reprospect.test.sass.instruction.constant import ConstantMatcher
from reprospect.test.sass.instruction.immediate import Immediate
from reprospect.test.sass.instruction.instruction import AnyMatcher, PatternBuilder
from reprospect.test.sass.instruction.register import Register, RegisterMatcher
from reprospect.tools.sass.decode import Decoder

from tests.cublas import CuBLAS
from tests.parameters import PARAMETERS, Parameters


class TestCuBLAS:
    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test_operands(self, workdir: pathlib.Path, parameters: Parameters, cmake_file_api) -> None:
        """
        For each supported architecture, pick one cubin from cuBLAS.
        From this cubin, pick a few functions.
        Loop over all instructions of each function, and match each operand.
        """
        cublas = CuBLAS(cmake_file_api=cmake_file_api)
        try:
            cuobjdump = cublas.cubin(arch=parameters.arch, cwd=workdir, sass=True)
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
            regex.compile(r'PR'),
            regex.compile(r'(SR_TID|SR_CTAID).(X|Y|Z)'),
            regex.compile(r'SR(X|Y|Z)'),
            regex.compile(r'SR_CgaCtaId|SR_LANEID|SR_LTMASK|SR_SWINHI'),
        ))

        # Pick 3 functions randomly.
        MAX_NUM_FUNCS: typing.Final[int] = len(cuobjdump.functions)
        for function in random.sample(tuple(cuobjdump.functions.keys()), k=min(len(cuobjdump.functions), MAX_NUM_FUNCS)):
            logging.info(f'Picked function {function} from {cuobjdump.file} for {parameters.arch}.')

            decoder = Decoder(code=cuobjdump.functions[function].code)

            for instruction in decoder.instructions:
                assert (matched := matcher.match(instruction)) is not None
                for operand in matched.operands:
                    if not any(mop.match(operand) is not None for mop in matchers_op):
                        logging.info(f'Instruction {instruction.instruction} has unmatched operand {operand}.')
                        pytest.fail()

# libcublas.2.sm_75.cubin for TURING75.
# HADD2 R21, -RZ.H0_H0, c[0x2] [0x0].F32

# libcublas.27.sm_80.cubin for AMPERE80
# HADD2.F32 R10, -RZ, c[0x0] [0x160].H0_H0

# libcublas.22.sm_86.cubin for AMPERE86
# HADD2.F32 R9, -RZ, c[0x0] [0x160].H0_H0

# libcublas.15.sm_80.cubin
# LDG.E R12, desc[UR4][R8.64]
