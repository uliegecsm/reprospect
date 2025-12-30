import sys
import typing

from reprospect.test.sass.instruction.address import AddressMatcher
from reprospect.test.sass.instruction.instruction import (
    ArchitectureAwarePatternMatcher,
    OpCode,
)
from reprospect.test.sass.instruction.memory import (
    ExtendBitsMethod,
    MemoryOp,
    MemorySpace,
)
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
    __slots__ = ('mop',)

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
        self.mop: typing.Final[MemoryOp] = MemoryOp(
            size=size,
            memory=MemorySpace(memory),
            extend=ExtendBitsMethod(extend) if extend is not None else None,
        )

        super().__init__(arch=arch)

    def _get_modifiers(self) -> typing.Iterable[str | int]:
        return filter(None, self.mop.get_modifiers() + (
            self.mop.get_sys(arch=self.arch),
        ))

    @override
    def _build_pattern(self) -> str:
        return (self.TEMPLATE_256 if self.mop.size is not None and self.mop.size == 256 else self.TEMPLATE).format(
            opcode=OpCode.mod(
                opcode=f'ST{self.mop.memory}',
                modifiers=self._get_modifiers(),
            ),
            address=PatternBuilder.groups(AddressMatcher.build_pattern(arch=self.arch, memory=self.mop.memory), groups=('operands', 'address')),
        )

class StoreGlobalMatcher(StoreMatcher):
    """
    Specialization of :py:class:`StoreMatcher` for global memory (``STG``).
    """
    def __init__(self, arch: NVIDIAArch, size: int | None = None, extend: ExtendBitsMethod | str | None = None) -> None:
        super().__init__(arch=arch, size=size, memory=MemorySpace.GLOBAL, extend=extend)
