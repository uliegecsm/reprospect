import re
import sys
import typing

from reprospect.test.sass.composite import instructions_are
from reprospect.test.sass.composite_impl import (
    OrderedInSequenceMatcher,
    SequenceMatcher,
)
from reprospect.test.sass.instruction import (
    InstructionMatch,
    RegisterMatch,
    RegisterMatcher,
)
from reprospect.test.sass.matchers import add_int128
from reprospect.tools.sass import ControlFlow, Decoder, Instruction

from examples.kokkos.atomic import add, cas, desul

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class RegisterMatchValidator(SequenceMatcher):
    """
    Validate that :py:attr:`matcher` uses :py:attr:`load_register`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('load_register', 'matcher', 'start_register_matcher')

    def __init__(self, matcher: add_int128.AddInt128Matcher, load: InstructionMatch) -> None:
        self.matcher: typing.Final[add_int128.AddInt128Matcher] = matcher
        """Inner matcher."""
        matched = RegisterMatcher().match(load.operands[0])
        assert matched is not None
        self.load_register: typing.Final[RegisterMatch] = matched
        """The register that must be used by :py:attr:`matcher`."""
        self.start_register_matcher: typing.Final[RegisterMatcher] = RegisterMatcher(rtype=self.load_register.rtype, index=self.load_register.index)

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        if (matched := self.matcher.match(instructions)) is not None:
            if self.start_register_matcher.match(matched[0].additional['start'][0]) is not None:
                return matched
        return None

    @override
    @property
    def next_index(self) -> int:
        return self.matcher.next_index

class AddInt128:
    """
    Addition of 2 :code:`__int128` that uses a specific set of registers.
    """
    def build(self, loads: typing.Collection[InstructionMatch]) -> OrderedInSequenceMatcher:
        if len(loads) != 1:
            raise RuntimeError(self)
        return instructions_are(RegisterMatchValidator(matcher=add_int128.AddInt128Matcher(), load=loads[0]))

class TestAtomicAddInt128(add.TestCase):
    """
    Tests for :code:`__int128`.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_atomic_add_int128'

    SIGNATURE_MATCHER: typing.ClassVar[re.Pattern[str]] = re.compile(
        r'AtomicAddFunctor<Kokkos::View<__int128\s*\*\s*, Kokkos::CudaSpace>>',
    )

    def test_lock_atomic_before_hopper90(self, decoder: Decoder) -> None:
        """
        This test proves that it uses the lock-based implementation.
        """
        matched = desul.LockBasedAtomicMatcher(
            arch=self.arch,
            operation=AddInt128(),
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
            operation=AddInt128(),
            size=128,
        ).match(cfg=ControlFlow.analyze(instructions=decoder.instructions))

        if self.arch.compute_capability.as_int >= 90:
            assert matched is not None
        else:
            assert matched is None
