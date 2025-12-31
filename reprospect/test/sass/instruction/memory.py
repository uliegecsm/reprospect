import dataclasses
import sys
import typing

from reprospect.tools.architecture import NVIDIAArch

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

class MemorySpace(StrEnum):
    """
    Allowed memory spaces.
    """
    GENERIC = ''
    GLOBAL = 'G'
    LOCAL = 'L'
    SHARED = 'S'

class ExtendBitsMethod(StrEnum):
    """
    How bits must be extended, see https://www.cs.fsu.edu/~hawkes/cda3101lects/chap4/extension.htm.
    """
    U = 'U'
    """Zero extension."""

    S = 'S'
    """Sign extension."""

def check_memory_instruction_word_size(*, size: int) -> None:
    """
    From https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses:

        Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes.
    """
    ALLOWABLE_SIZES: typing.Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32) # pylint: disable=invalid-name
    if size not in ALLOWABLE_SIZES:
        raise RuntimeError(f'{size} is not an allowable memory instruction word size ({ALLOWABLE_SIZES} (in bytes)).')

@dataclasses.dataclass(frozen=True, slots=True)
class MemoryOp:
    size: int | None
    memory: MemorySpace
    extend: ExtendBitsMethod | None

    def __post_init__(self) -> None:
        if self.size is not None:
            check_memory_instruction_word_size(size=self.size // 8)

    def get_size(self) -> int | str | None:
        if self.size is not None and self.size < 32:
            return f'{self.extend}{self.size}'
        if self.size is not None and self.size > 32:
            return self.size
        return None

    def get_enl2(self) -> str | None:
        if self.size is not None and self.size == 256:
            return 'ENL2'
        return None

    def get_modifiers(self) -> tuple[int | str | None, ...]:
        return (
            'E',
            self.get_enl2(),
            self.get_size(),
        )

    @staticmethod
    def get_sys(*, arch: NVIDIAArch) -> str | None:
        if arch.compute_capability.as_int in {70, 75}:
            return 'SYS'
        return None
