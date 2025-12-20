import logging
import re
import sys
import typing

from reprospect.test.features import Memory
from reprospect.test.sass.composite import (
    instruction_is,
    interleaved_instructions_are,
    unordered_interleaved_instructions_are,
)
from reprospect.test.sass.composite_impl import (
    SequenceMatcher,
    UnorderedInSequenceMatcher,
)
from reprospect.test.sass.instruction import (
    InstructionMatch,
    LoadGlobalMatcher,
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
    RegisterMatcher,
    StoreGlobalMatcher,
)
from reprospect.test.sass.instruction.constant import Constant
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass import Decoder

from examples.kokkos.atomic import add, desul

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Load256Matcher:
    def __init__(self, arch: NVIDIAArch) -> None:
        self.arch: typing.Final[NVIDIAArch] = arch

    def build(self) -> SequenceMatcher:
        match Memory(arch=self.arch).max_transaction_size:
            case 16:
                return interleaved_instructions_are(
                    LoadGlobalMatcher(arch=self.arch, size=128, readonly=False),
                    LoadGlobalMatcher(arch=self.arch, size=128, readonly=False),
                )
            case 32:
                return instruction_is(LoadGlobalMatcher(arch=self.arch, size=256, readonly=False)).times(num=1)
            case _:
                raise ValueError

class Store256Matcher:
    def __init__(self, arch: NVIDIAArch) -> None:
        self.arch: typing.Final[NVIDIAArch] = arch

    def build(self) -> SequenceMatcher:
        match Memory(arch=self.arch).max_transaction_size:
            case 16:
                return interleaved_instructions_are(
                    StoreGlobalMatcher(arch=self.arch, size=128),
                    StoreGlobalMatcher(arch=self.arch, size=128),
                )
            case 32:
                return instruction_is(StoreGlobalMatcher(arch=self.arch, size=256)).times(num=1)
            case _:
                raise ValueError

class AddDouble4:
    """
    Addition of 2 :code:`double4` (whatever the alignment).
    """
    def __init__(self, arch: NVIDIAArch) -> None:
        self.arch: typing.Final[NVIDIAArch] = arch

    def build(self, loads: typing.Collection[InstructionMatch]) -> UnorderedInSequenceMatcher:
        match Memory(arch=self.arch).max_transaction_size:
            case 16:
                assert len(loads) == 2
                registers = (loads[0].operands[0], loads[1].operands[0])
            case 32:
                assert len(loads) == 1
                registers = tuple(reversed(loads[0].operands[0:2]))
            case _:
                raise ValueError

        matchers: list[OpcodeModsWithOperandsMatcher] = []

        for register in registers:
            matched = RegisterMatcher(special=False).match(register)
            assert matched is not None and matched.index is not None

            logging.info(f'Load register {matched}.')

            matchers.append(OpcodeModsWithOperandsMatcher(opcode='DADD',
                operands=(
                    register, register,
                    PatternBuilder.any(PatternBuilder.UREG, Constant.ADDRESS),
                ),
            ))
            matchers.append(OpcodeModsWithOperandsMatcher(opcode='DADD',
                operands=(
                    f'{matched.rtype}{matched.index + 2}',
                    f'{matched.rtype}{matched.index + 2}',
                    PatternBuilder.any(PatternBuilder.UREG, Constant.ADDRESS),
                ),
            ))

        return unordered_interleaved_instructions_are(*matchers)

class TestAtomicAddDouble256(add.TestCase):
    """
    Verify that :code:`Kokkos::atomic_add` for :code:`double4` maps to
    the `desul` lock-based array implementation (whatever the alignment).
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_atomic_add_double256'

    SIGNATURE_MATCHER: typing.ClassVar[re.Pattern[str]] = re.compile(
        r'AtomicAddFunctor<Kokkos::View<reprospect::examples::kokkos::atomic::Double4Aligned32\s*\*\s*, Kokkos::CudaSpace>>',
    )

    def test_lock_atomic(self, decoder: Decoder) -> None:
        """
        This test proves that it uses the lock-based implementation.
        """
        desul.LockBasedAtomicMatcher(
            arch=self.arch,
            load=Load256Matcher(arch=self.arch).build(),
            operation=AddDouble4(arch=self.arch),
            store=Store256Matcher(arch=self.arch).build(),
            compiler_id=self.toolchains['CUDA']['compiler']['id'],
        ).assert_matches(instructions=decoder.instructions)
