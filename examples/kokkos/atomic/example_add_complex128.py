import re
import sys
import typing

from reprospect.test.sass.composite import unordered_interleaved_instructions_are
from reprospect.test.sass.composite_impl import UnorderedInSequenceMatcher
from reprospect.test.sass.instruction import (
    InstructionMatch,
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
    RegisterMatcher,
)
from reprospect.tools.sass import Decoder

from examples.kokkos.atomic import add, desul

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class AddKokkosComplexDouble:
    """
    Addition of 2 :code:`Kokkos::complex<double>`.

    1. real parts
    2. imaginary parts
    3. possibly with NOP
    """
    def build(self, loads : typing.Collection[InstructionMatch]) -> UnorderedInSequenceMatcher:
        if len(loads) != 1:
            raise RuntimeError(self)
        load_register = loads[0].operands[0]

        parsed = RegisterMatcher(special = False).match(load_register)
        assert parsed is not None
        assert parsed.index is not None

        dadd_real_reg = load_register
        dadd_imag_reg = f'{parsed.rtype}{parsed.index + 2}'

        matcher_dadd_real = OpcodeModsWithOperandsMatcher(opcode = 'DADD',
            operands = (
                dadd_real_reg, dadd_real_reg,
                PatternBuilder.any(PatternBuilder.UREG, PatternBuilder.CONSTANT),
            ),
        )
        matcher_dadd_imag = OpcodeModsWithOperandsMatcher(opcode = 'DADD',
            operands = (
                dadd_imag_reg, dadd_imag_reg,
                PatternBuilder.any(PatternBuilder.UREG, PatternBuilder.CONSTANT),
            ),
        )

        return unordered_interleaved_instructions_are(matcher_dadd_real, matcher_dadd_imag)

class TestAtomicAddComplex128(add.TestCase):
    """
    Verify that :code:`Kokkos::atomic_add` for :code:`Kokkos::complex<double>` maps to
    the `desul` lock-based array implementation.

    Although :code:`Kokkos::complex<double>` meets the requirements for 128-bit CAS,
    the current embedded `desul` version does not support it.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_atomic_add_complex128'

    SIGNATURE_MATCHER : typing.ClassVar[re.Pattern[str]] = re.compile(
        r'AtomicAddFunctor<Kokkos::View<Kokkos::complex<double>\s*\*\s*, Kokkos::CudaSpace>>',
    )

    def test_lock_based_atomic(self, decoder : Decoder) -> None:
        """
        This test proves that it uses the lock-based atomic by looking for an instruction sequence pattern.
        """
        desul.LockBasedAtomicMatcher(
            arch = self.arch,
            operation = AddKokkosComplexDouble(),
            compiler_id = self.toolchains['CUDA']['compiler']['id'],
        ).assert_matches(instructions = decoder.instructions)
