import sys
import typing

from reprospect.test.sass.instruction.address import AddressMatcher
from reprospect.test.sass.instruction.constant import ConstantMatcher
from reprospect.test.sass.instruction.instruction import (
    ArchitectureAwarePatternMatcher,
    OpCode,
    PatternMatcher,
    ZeroOrOne,
)
from reprospect.test.sass.instruction.memory import (
    ExtendBitsMethod,
    MemoryOp,
    MemorySpace,
    check_memory_instruction_word_size,
)
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class LoadMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for load instructions, such as::

        LDG.E R2, desc[UR6][R2.64]
        LD.E.64 R2, R4.64

    Starting from `BLACKWELL`, 256-bit load instructions are available, such as::

        LDG.E.ENL2.256.CONSTANT R12, R8, desc[UR4][R2.64]

    References:

    * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldg#global-memory-5-x
    * https://github.com/AdaptiveCpp/AdaptiveCpp/issues/848
    * https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsglobal.htm
    * https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/#updated_vector_types
    """
    __slots__ = ('cache', 'mop')

    TEMPLATE:     typing.Final[str] = f'{{opcode}} {Register.reg()}, {{address}}'
    TEMPLATE_256: typing.Final[str] = f'{{opcode}} {Register.reg()}, {Register.reg()}, {{address}}'

    def __init__(self,
        arch: NVIDIAArch,
        *,
        size: int | None = None,
        readonly: bool | None = None,
        memory: MemorySpace | str = MemorySpace.GLOBAL,
        extend: ExtendBitsMethod | str | None = None,
    ) -> None:
        """
        :param size: Optional bit size (*e.g.*, 32, 64, 128).
        :param readonly: Whether to append ``.CONSTANT`` modifier. If `None`, the modifier is matched optionally.
        """
        self.mop: typing.Final[MemoryOp] = MemoryOp(
            size=size,
            memory=MemorySpace(memory),
            extend=ExtendBitsMethod(extend) if extend is not None else None,
        )

        self.cache: str | ZeroOrOne | None = None if readonly is False else ('CONSTANT' if readonly is True else ZeroOrOne('CONSTANT'))

        super().__init__(arch=arch)

    def _get_modifiers(self) -> typing.Iterable[str | int | ZeroOrOne]:
        return filter(None, self.mop.get_modifiers() + (
            self.cache,
            self.mop.get_sys(arch=self.arch),
        ))

    @override
    def _build_pattern(self) -> str:
        return (self.TEMPLATE_256 if self.mop.size is not None and self.mop.size == 256 else self.TEMPLATE).format(
            opcode=OpCode.mod(
                opcode=f'LD{self.mop.memory.value}',
                modifiers=self._get_modifiers(),
            ),
            address=PatternBuilder.groups(AddressMatcher.build_pattern(arch=self.arch, memory=self.mop.memory), groups=('operands', 'address')),
        )

class LoadGlobalMatcher(LoadMatcher):
    """
    Specialization of :py:class:`LoadMatcher` for global memory (``LDG``).
    """
    def __init__(self, arch: NVIDIAArch, *, size: int | None = None, readonly: bool | None = None, extend: ExtendBitsMethod | str | None = None) -> None:
        super().__init__(arch=arch, size=size, readonly=readonly, memory=MemorySpace.GLOBAL, extend=extend)

class LoadConstantMatcher(PatternMatcher):
    """
    Matcher for constant load (``LDC``) instructions, such as::

        LDC.64 R2, c[0x0][0x388]
        LDC R4, c[0x3][R0]
        LDCU UR4, c[0x3][UR0]
    """
    CONSTANT: typing.Final[str] = ConstantMatcher.build_pattern(captured=True, capture_bank=True, capture_offset=True)

    TEMPLATE: typing.Final[str] = f'{{opcode}} {{dest}}, {CONSTANT}'

    def __init__(self, *, uniform: bool | None = None, size: int | None = None) -> None:
        """
        :param size: Optional bit size (*e.g.*, 32, 64, 128).
        :param uniform: Optionally require uniformness.
        """
        if size is not None:
            check_memory_instruction_word_size(size=size // 8)

        if uniform is None:
            opcode = PatternBuilder.any('LDC', 'LDCU')
            dest   = PatternBuilder.any(Register.REG, Register.UREG)
        elif uniform is True:
            opcode = 'LDCU'
            dest   = Register.UREG
        else:
            opcode = 'LDC'
            dest   = Register.REG

        super().__init__(pattern=self.TEMPLATE.format(
            opcode=OpCode.mod(
                opcode=opcode,
                modifiers=(size,) if size else (),
            ),
            dest=PatternBuilder.group(dest, group='operands'),
        ))
