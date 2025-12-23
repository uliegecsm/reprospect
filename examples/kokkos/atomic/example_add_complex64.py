import re
import sys
import typing

from reprospect.test.sass.composite import (
    instruction_is,
    unordered_interleaved_instructions_are,
)
from reprospect.test.sass.composite_impl import UnorderedInterleavedInSequenceMatcher
from reprospect.test.sass.instruction import (
    Fp32AddMatcher,
    InstructionMatch,
    RegisterMatcher,
)
from reprospect.tools.sass import ControlFlow, Decoder

from examples.kokkos.atomic import add, cas

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class AddComplex64:
    """
    Addition of two 64-bit complex values.
    """
    def build(self, loads: typing.Collection[InstructionMatch]) -> UnorderedInterleavedInSequenceMatcher:
        assert len(loads) == 1

        assert (load_reg := RegisterMatcher(special=False).match(reg=loads[0].operands[0])) is not None

        reg_real = f'{load_reg.rtype}{load_reg.index}'
        reg_imag = f'{load_reg.rtype}{load_reg.index + 1}'

        return unordered_interleaved_instructions_are(
            instruction_is(Fp32AddMatcher()).with_operand(index=1, operand=reg_real),
            instruction_is(Fp32AddMatcher()).with_operand(index=1, operand=reg_imag),
        )

class TestAtomicAddComplex64(add.TestCase):
    """
    Tests for :code:`Kokkos::complex<float>`.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_atomic_add_complex64'

    SIGNATURE_MATCHER: typing.ClassVar[re.Pattern[str]] = re.compile(
        r'AtomicAddFunctor<Kokkos::View<Kokkos::complex<float>\s*\*\s*, Kokkos::CudaSpace>>',
    )

    def test_cas_atomic(self, decoder: Decoder) -> None:
        """
        This test proves that it uses the CAS-based implementation.
        """
        assert cas.AtomicCAS(
            arch=self.arch,
            operation=AddComplex64(),
            size=64,
        ).match(cfg=ControlFlow.analyze(instructions=decoder.instructions)) is not None
