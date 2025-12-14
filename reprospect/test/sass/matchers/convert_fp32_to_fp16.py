import sys
import typing

from reprospect.test.sass.instruction import (
    InstructionMatch,
    InstructionMatcher,
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
)
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass.decode import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class ConvertFp32ToFp16(InstructionMatcher):
    """
    Convert 32-bit floating-point value to 16-bit floating-point value.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, arch: NVIDIAArch, *,
        dst : str = PatternBuilder.REG,
        src : str = PatternBuilder.REG,
    ) -> None:
        """
        :param src: 32-bit floating-point value.
        :param dst: 16-bit floating-point value.
        """
        match arch.compute_capability.as_int:
            case 70 | 75:
                matcher = OpcodeModsWithOperandsMatcher(opcode='F2F', modifiers=('F16', 'F32'), operands=(dst, src))
            case 80 | 86:
                matcher = OpcodeModsWithOperandsMatcher(opcode='F2FP', modifiers=('PACK_AB',), operands=(dst, 'RZ', src))
            case _:
                matcher = OpcodeModsWithOperandsMatcher(opcode='F2FP', modifiers=('F16', 'F32', 'PACK_AB'), operands=(dst, 'RZ', src))
        self.matcher : typing.Final[OpcodeModsWithOperandsMatcher] = matcher

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        return self.matcher.match(inst=inst)
