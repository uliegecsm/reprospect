"""
:code:`Kokkos` math function for half-precision types
must use specialized FP16 hardware units if available (rather than the FP32 ones)
to achieve peak performance.
Therefore, traditional output-correctness tests for the half-precision types are
not sufficient.

:code:`Kokkos` relies on a complex implementation strategy with many macros
for providing math function overload resolution.
For half-precision types, :code:`Kokkos` provides a
`templated fallback <https://github.com/kokkos/kokkos/blob/f4de0debe3053bb1babc084b5731ad938952f563/core/src/impl/Kokkos_Half_MathematicalFunctions.hpp#L110-L114>`_
that casts to :code:`float`, and the CUDA backend provides a
`non-template overload <https://github.com/kokkos/kokkos/blob/f4de0debe3053bb1babc084b5731ad938952f563/core/src/Cuda/Kokkos_Cuda_Half_MathematicalFunctions.hpp#L19-L21>`_
that uses the CUDA intrinsics.
According to the
`best viable function rules <https://en.cppreference.com/w/cpp/language/overload_resolution.html#Best_viable_function>`_,
the non-template overload is selected.

Manual inspection of the source code to verify that the correct implementation path is maintained
as the supported architecture set and code base evolve is challenging, and requires substantial review effort.

:py:class:`TestMax` shows how implementation correctness can be confirmed through automated SASS analysis for :code:`Kokkos::fmax`
for which a CUDA intrinsic :code:`__hmax` exists.

References:

* https://github.com/kokkos/kokkos/pull/8719
"""


import enum
import logging
import pathlib
import re
import sys
import typing

import pytest

from reprospect.test import CMakeAwareTestCase
from reprospect.test.sass.composite import (
    instructions_are,
    instructions_contain,
    instruction_is,
)
from reprospect.test.sass.controlflow.block import BlockMatcher
from reprospect.test.sass.instruction import (
    InstructionMatch,
    LoadGlobalMatcher,
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
    StoreGlobalMatcher,
)
from reprospect.test.sass.instruction.half import Fp16AddMatcher, Fp16MinMaxMatcher
from reprospect.test.sass.matchers.convert_fp32_to_fp16 import ConvertFp32ToFp16
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass.controlflow import BasicBlock
from reprospect.tools.sass import ControlFlow, Decoder, Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Method(enum.IntEnum):
    CUDA_HMAX = 0
    """Use the CUDA intrinsic :code:`__hmax`."""
    FMAX = 1
    """Use :code:`fmax(float, float)`."""
    KOKKOS_FMAX = 2
    """Use :code:`Kokkos::fmax(Kokkos::Experimental::half_t, Kokkos::Experimental::half_t)`."""

class TestMax(CMakeAwareTestCase):
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_half_math'

    SIGNATURE_TEMPLATE : typing.Final[str] = (
        r'void Kokkos::Impl::cuda_parallel_launch_local_memory<'
        r'Kokkos::Impl::ParallelFor<'
        r'reprospect::examples::kokkos::half::FunctorMax<'
        r'\(reprospect::examples::kokkos::half::Method\){method}, '
        r'Kokkos::View<Kokkos::Experimental::Impl::floating_point_wrapper<__half>\s*\*, Kokkos::Cuda>>, Kokkos::RangePolicy<>, Kokkos::Cuda>>'
    )

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'{self.get_target_name()}.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope = 'class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(
            file = self.executable,
            arch = self.arch,
            sass = True, cwd = self.cwd,
            cubin = self.cubin.name,
            demangler = self.demangler,
        )[0]

    @pytest.fixture(scope = 'class')
    def decoder(self, cuobjdump : CuObjDump) -> dict[Method, Decoder]:
        decoder : dict[Method, Decoder] = {}
        for method in Method:
            matcher = re.compile(self.SIGNATURE_TEMPLATE.format(method=method.value))
            [sig] = (sig for sig in cuobjdump.functions if matcher.search(sig) is not None)
            decoder[method] = Decoder(code = cuobjdump.functions[sig].code)
        return decoder

    def test_cuda_hmax(self, decoder : dict[Method, Decoder]) -> None:
        """
        Check SASS code for :py:attr:`Method.CUDA_HMAX`.

        .. note::

            Before compute capability 8.0, the intrinsic :code:`__hmax` generates FP32 instructions.
        """
        if self.arch.compute_capability >= 80:
            self.match_fp16(instructions=decoder[Method.CUDA_HMAX].instructions)
        else:
            self.match_fp32(instructions=decoder[Method.CUDA_HMAX].instructions)

    def test_kokkos_fmax(self, decoder : dict[Method, Decoder]) -> None:
        """
        Check SASS code for :py:attr:`Method.KOKKOS_FMAX`.

        .. note::

            It always leads to the exact same SASS code as :py:attr:`Method.CUDA_HMAX`, thus confirming that
            it is implemented correctly.
        """
        assert decoder[Method.KOKKOS_FMAX].instructions == decoder[Method.CUDA_HMAX].instructions

    def test_fmax(self, decoder : dict[Method, Decoder]) -> None:
        """
        Check SASS code for :py:attr:`Method.FMAX`.
        """
        self.match_fp32(instructions=decoder[Method.FMAX].instructions)


    def match_block_and_loads(self, *,
        instructions : typing.Sequence[Instruction],
    ) -> tuple[BasicBlock, int, InstructionMatch, InstructionMatch]:
        """
        Find the block with two 16-bit loads, such as::

            LDG.E.U16 R2, [R2.64]
            LDG.E.U16 R5, [R4.64]
        """
        matcher_ldg = instructions_contain(instructions_are(
            LoadGlobalMatcher(arch = self.arch, size = 16, extend = 'U', readonly = False),
            LoadGlobalMatcher(arch = self.arch, size = 16, extend = 'U', readonly = False),
        ))
        block, matched_ldg = BlockMatcher(matcher_ldg).assert_matches(cfg=ControlFlow.analyze(instructions=instructions))
        logging.info(matched_ldg)
        assert len(matched_ldg) == 2

        return block, matcher_ldg.next_index, matched_ldg[0], matched_ldg[1]

    def match_store(self, src : str, instructions : typing.Sequence[Instruction]) -> None:
        instructions_contain(instruction_is(StoreGlobalMatcher(
            arch = self.arch, size = 16, extend = 'U',
        )).with_operand(index = -1, operand = src)).assert_matches(instructions=instructions)

    def match_fp16_to_fp32(self, *,
        src_a : str, src_b : str,
        instructions : typing.Sequence[Instruction],
    ) -> tuple[int, InstructionMatch, InstructionMatch]:
        """
        Conversion from FP16 to FP32.
        """
        matcher_hadd2_a = instructions_contain(instruction_is(Fp16AddMatcher(packed=None)).with_modifier('F32').with_operand(f'{src_a}.H0_H0'))
        [matched_hadd2_a] = matcher_hadd2_a.assert_matches(instructions=instructions)
        logging.info(matched_hadd2_a)
        offset = matcher_hadd2_a.next_index

        matcher_hadd2_b = instructions_contain(instruction_is(Fp16AddMatcher(packed=None)).with_operand(f'{src_b}.H0_H0'))
        [matched_hadd2_b] = matcher_hadd2_b.assert_matches(instructions=instructions[offset:])
        logging.info(matched_hadd2_b)
        offset += matcher_hadd2_b.next_index

        return offset, matched_hadd2_a, matched_hadd2_b

    def match_fp32_to_fp16(self, *,
        src : str, instructions : typing.Sequence[Instruction],
    ) -> tuple[int, InstructionMatch]:
        """
        Convert from FP32 to FP16.
        """
        matcher = instructions_contain(ConvertFp32ToFp16(arch=self.arch, src=src))
        [matched] = matcher.assert_matches(instructions=instructions)
        logging.info(matched)

        return matcher.next_index, matched

    def match_fp16(self, instructions : typing.Sequence[Instruction]) -> None:
        """
        Typically::

            LDG.E.U16 R2, desc[UR6][R2.64]
            LDG.E.U16 R5, desc[UR6][R4.64]
            ...
            HMNMX2 R5, R2.H0_H0, R5.H0_H0, !PT
            ...
            STG.E.U16 desc[UR6][R6.64], R5
        """
        block, offset, matched_ldg_a, matched_ldg_b = self.match_block_and_loads(instructions=instructions)

        matcher_hmnmx2 = instructions_contain(instruction_is(Fp16MinMaxMatcher(pmax=True)).with_operand(
            index=1, operand=f'{matched_ldg_a.operands[0]}.H0_H0',
        ).with_operand(
            index=2, operand=f'{matched_ldg_b.operands[0]}.H0_H0',
        ))
        [matched_hmnmx2] = matcher_hmnmx2.assert_matches(instructions=block.instructions[offset:])
        offset += matcher_hmnmx2.next_index

        self.match_store(src=matched_hmnmx2.operands[0], instructions=block.instructions[offset:])

    def match_fp32(self, instructions : typing.Sequence[Instruction]) -> None:
        """
        Typically::

            LDG.E.U16 R2, desc[UR6][R2.64]
            LDG.E.U16 R4, desc[UR6][R4.64]
            ...
            HADD2.F32 R6, -RZ, R2.H0_H0
            HADD2.F32 R7, -RZ, R4.H0_H0
            FMNMX R6, R6, R7, !PT
            F2FP.F16.F32.PACK_AB R3, RZ, R6
            ...
            STG.E.U16 desc[UR6][R6.64], R3
        """
        block, offset, matched_ldg_a, matched_ldg_b = self.match_block_and_loads(instructions=instructions)

        advanced, matched_hadd2_a, matched_hadd2_b = self.match_fp16_to_fp32(
            src_a=matched_ldg_a.operands[0],
            src_b=matched_ldg_b.operands[0],
            instructions=block.instructions[offset:],
        )
        offset += advanced

        # Take the max.
        matcher_fmnmx = instructions_contain(OpcodeModsWithOperandsMatcher(opcode='FMNMX', operands=(
            PatternBuilder.REG,
            matched_hadd2_a.operands[0], matched_hadd2_b.operands[0], '!PT',
        )))
        [matched_fmnmx] = matcher_fmnmx.assert_matches(instructions=block.instructions[offset:])
        logging.info(matched_fmnmx)

        advanced, matched = self.match_fp32_to_fp16(src=matched_fmnmx.operands[0], instructions=block.instructions[offset:])
        offset += advanced

        self.match_store(src=matched.operands[0], instructions=block.instructions[offset:])
