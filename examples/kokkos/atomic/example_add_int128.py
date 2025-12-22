import re
import sys
import typing

from reprospect.test.sass.instruction import (
    InstructionMatch,
    RegisterMatcher,
)
from reprospect.test.sass.matchers import add_int128
from reprospect.test.sass.matchers.cas import AtomicCASMatcher
from reprospect.tools.sass import ControlFlow, Decoder

from examples.kokkos.atomic import add, desul

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class AddInt128:
    """
    Addition of 2 :code:`__int128` that uses a specific set of registers.
    """
    def build(self, loads: typing.Collection[InstructionMatch] | None = None) -> add_int128.AddInt128Matcher:
        if loads is not None:
            assert len(loads) == 1

            assert (matched := RegisterMatcher().match(loads[0].operands[0])) is not None

            return add_int128.AddInt128Matcher(
                start=RegisterMatcher.build_pattern(
                    rtype=matched.rtype,
                    index=matched.index,
                    reuse=None,
                    math=False,
                ),
            )
        return add_int128.AddInt128Matcher()

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
        matched = AtomicCASMatcher(
            arch=self.arch,
            operation=AddInt128(),
            size=128,
        ).match(cfg=ControlFlow.analyze(instructions=decoder.instructions))

        if self.arch.compute_capability.as_int >= 90:
            assert matched is not None
        else:
            assert matched is None
