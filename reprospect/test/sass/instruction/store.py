import sys
import typing

from reprospect.test.sass.instruction.address import AddressMatcher
from reprospect.test.sass.instruction.instruction import (
    ArchitectureAwarePatternMatcher,
    ExtendBitsMethod,
    OpCode,
    check_memory_instruction_word_size,
    memory_op_get_size,
)
from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class StoreMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for global store instructions, such as::

        STG.E desc[UR6][R6.64], R15
        ST.E.64 R4.64, R2

    Starting from `BLACKWELL`, 256-bit store instructions are available, such as::

        STG.E.ENL2.256 desc[UR4][R4.64], R8, R12
    """
    __slots__ = ('extend', 'memory', 'size')

    TEMPLATE:     typing.Final[str] = f'{{opcode}} {{address}}, {Register.reg()}'
    TEMPLATE_256: typing.Final[str] = f'{{opcode}} {{address}}, {Register.reg()}, {Register.reg()}'

    def __init__(self,
        arch: NVIDIAArch,
        size: int | None = None,
        memory: MemorySpace | str = MemorySpace.GLOBAL,
        extend: ExtendBitsMethod | str | None = None,
    ) -> None:
        """
        :param size: Optional bit size (*e.g.*, 32, 64, 128).
        """
        if size is not None:
            check_memory_instruction_word_size(size=size // 8)

        self.size: typing.Final[int | None] = size
        self.memory: typing.Final[MemorySpace] = MemorySpace(memory)
        self.extend: typing.Final[ExtendBitsMethod | None] = ExtendBitsMethod(extend) if extend is not None else None
        super().__init__(arch=arch)

    def _get_modifiers(self) -> typing.Iterable[str | int]:
        return filter(None, (
            'E',
            *(('ENL2',) if self.size is not None and self.size == 256 else ()),
            memory_op_get_size(size=self.size, extend=self.extend),
            *(('SYS',) if self.arch.compute_capability.as_int in {70, 75} else ()),
        ))

    @override
    def _build_pattern(self) -> str:
        return (self.TEMPLATE_256 if self.size is not None and self.size == 256 else self.TEMPLATE).format(
            opcode=OpCode.mod(
                opcode=f'ST{self.memory}',
                modifiers=self._get_modifiers(),
            ),
            address=PatternBuilder.groups(AddressMatcher.build_pattern(arch=self.arch, memory=self.memory), groups=('operands', 'address')),
        )

class StoreGlobalMatcher(StoreMatcher):
    """
    Specialization of :py:class:`StoreMatcher` for global memory (``STG``).
    """
    def __init__(self, arch: NVIDIAArch, size: int | None = None, extend: ExtendBitsMethod | str | None = None) -> None:
        super().__init__(arch=arch, size=size, memory=MemorySpace.GLOBAL, extend=extend)
