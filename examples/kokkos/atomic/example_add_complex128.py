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
from reprospect.test.sass.instruction.constant import Constant
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.sass import ControlFlow, Decoder

from examples.kokkos.atomic import add, cas, desul

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class AddComplex128:
    """
    Addition of two 128-bit complex values.

    1. real parts
    2. imaginary parts
    3. possibly with NOP
    """
    def build(self, loads: typing.Collection[InstructionMatch]) -> UnorderedInSequenceMatcher:
        if len(loads) != 1:
            raise RuntimeError(self)
        load_register = loads[0].operands[0]

        parsed = RegisterMatcher(special=False).match(load_register)
        assert parsed is not None
        assert parsed.index is not None

        dadd_real_reg = load_register
        dadd_imag_reg = f'{parsed.rtype}{parsed.index + 2}'

        matcher_dadd_real = OpcodeModsWithOperandsMatcher(opcode='DADD',
            operands=(
                Register.REG, dadd_real_reg, PatternBuilder.any(Register.UREG, Constant.ADDRESS),
            ),
        )
        matcher_dadd_imag = OpcodeModsWithOperandsMatcher(opcode='DADD',
            operands=(
                Register.REG, dadd_imag_reg, PatternBuilder.any(Register.UREG, Constant.ADDRESS),
            ),
        )

        return unordered_interleaved_instructions_are(matcher_dadd_real, matcher_dadd_imag)

class TestAtomicAddComplex128(add.TestCase):
    """
    Tests for :code:`Kokkos::complex<double>`.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_atomic_add_complex128'

    SIGNATURE_MATCHER: typing.ClassVar[re.Pattern[str]] = re.compile(
        r'AtomicAddFunctor<Kokkos::View<Kokkos::complex<double>\s*\*\s*, Kokkos::CudaSpace>>',
    )

    def test_lock_atomic_before_hopper90(self, decoder: Decoder) -> None:
        """
        This test proves that it uses the lock-based implementation.
        """
        matched = desul.LockBasedAtomicMatcher(
            arch=self.arch,
            operation=AddComplex128(),
            compiler_id=self.toolchains['CUDA']['compiler']['id'],
        ).match(instructions=decoder.instructions)

        if self.arch.compute_capability.as_int >= 90:
            assert matched is None
        else:
            assert matched is not None

    def test_cas_atomic_as_of_hopper90(self, decoder: Decoder) -> None:
        """
        This test proves that it uses the CAS-based implementation.
        """
        matched = cas.AtomicCAS(
            arch=self.arch,
            operation=AddComplex128(),
            size=128,
        ).match(cfg=ControlFlow.analyze(instructions=decoder.instructions))

        if self.arch.compute_capability.as_int >= 90:
            assert matched is not None
        else:
            assert matched is None
