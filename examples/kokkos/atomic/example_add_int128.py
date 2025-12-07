import re
import sys
import typing

from reprospect.test.sass.composite      import instructions_are
from reprospect.test.sass.composite_impl import OrderedInSequenceMatcher, SequenceMatcher
from reprospect.test.sass.instruction    import AtomicMatcher, \
                                                InstructionMatch, \
                                                ReductionMatcher, \
                                                RegisterMatch, \
                                                RegisterMatcher
from reprospect.test.sass.matchers       import add_int128
from reprospect.tools.sass               import Decoder, Instruction

from examples.kokkos.atomic import add, desul

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class RegisterMatchValidator(SequenceMatcher):
    """
    Validate that :py:attr:`matcher` uses :py:attr:`load_register`.
    """
    def __init__(self, matcher : add_int128.AddInt128, load : InstructionMatch) -> None:
        self.matcher : typing.Final[add_int128.AddInt128] = matcher
        """Inner matcher."""
        matched = RegisterMatcher().match(load.operands[0])
        assert matched is not None
        self.load_register : typing.Final[RegisterMatch] = matched
        """The register that must be used by :py:attr:`matcher`."""
        self.start_register_matcher : typing.Final[RegisterMatcher] = RegisterMatcher(rtype = self.load_register.rtype, index = self.load_register.index)

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        if (matched := self.matcher.match(instructions)) is not None:
            if self.start_register_matcher.match(matched[0].additional['start'][0]) is not None:
                return matched
        return None

class AddInt128:
    """
    Addition of 2 :code:`__int128` that uses a specific set of registers.
    """
    def build(self, load : InstructionMatch) -> OrderedInSequenceMatcher:
        return instructions_are(RegisterMatchValidator(matcher = add_int128.AddInt128(), load = load))

class TestAtomicAddInt128(add.TestCase):
    """
    Verify that :code:`Kokkos::atomic_add` for :code:`__int128` maps to
    the `desul` lock-based array implementation.

    Although :code:`__int128` meets the requirements for 128-bit CAS,
    the current embedded `desul` version does not support it.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_atomic_add_int128'

    SIGNATURE_MATCHER : typing.ClassVar[re.Pattern[str]] = re.compile(
        r'AtomicAddFunctor<Kokkos::View<__int128\s*\*\s*, Kokkos::CudaSpace>>'
    )

    def test_no_atomic_add_128(self, decoder : Decoder) -> None:
        """
        There is no match for an atomic add of size 128-bits.
        """
        matcher_atom = AtomicMatcher   (operation = 'ADD', dtype = ('S', 128), scope = 'DEVICE', consistency = 'STRONG', arch = self.arch)
        matcher_red  = ReductionMatcher(operation = 'ADD', dtype = ('S', 128), scope = 'DEVICE', consistency = 'STRONG', arch = self.arch)

        assert not any(matcher_atom.match(inst) for inst in decoder.instructions)
        assert not any(matcher_red .match(inst) for inst in decoder.instructions)

    def test_lock_based_atomic(self, decoder : Decoder) -> None:
        """
        This test proves that it uses the lock-based atomic by looking for a multi-instruction pattern.
        """
        desul.LockBasedAtomicMatcher(
            arch = self.arch,
            operation = AddInt128(),
            compiler_id = self.toolchains['CUDA']['compiler']['id'],
        ).assert_matches(instructions = decoder.instructions)
