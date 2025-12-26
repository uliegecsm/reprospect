import sys
import typing

from reprospect.test.sass.instruction import (
    InstructionMatch,
    InstructionMatcher,
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
)
from reprospect.test.sass.instruction.operand import Operand
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.sass.decode import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class Move32Matcher(InstructionMatcher):
    """
    Match the *move* of a 32-bit source to a 32-bit destination.

    It can be achieved using one of the following instructions::

        MOV <dst>, <src>
        IMAD.MOV.U32 <dst>, RZ, RZ, <src>

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('imad', 'mov')

    def __init__(self, *,
        src: str | None = None,
        dst: str | None = None,
    ) -> None:
        self.mov: typing.Final[OpcodeModsWithOperandsMatcher] = OpcodeModsWithOperandsMatcher(
            opcode='MOV', operands=(
                PatternBuilder.group(dst or Register.REG, group='dst'),
                PatternBuilder.group(src or Operand.OPERAND, group='src'),
            ),
        )

        self.imad: typing.Final[OpcodeModsWithOperandsMatcher] = OpcodeModsWithOperandsMatcher(
            opcode='IMAD', modifiers=('MOV', 'U32'), operands=(
                PatternBuilder.group(dst or Register.REG, group='dst'),
                'RZ', 'RZ',
                PatternBuilder.group(src or Operand.OPERAND, group='src'),
            ),
        )

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        if (matched := self.mov.match(inst=inst)) is not None:
            return matched
        if (matched := self.imad.match(inst=inst)) is not None:
            return matched
        return None
