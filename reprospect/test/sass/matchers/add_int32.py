import sys
import typing

from reprospect.test.sass.instruction import (
    InstructionMatch,
    InstructionMatcher,
    OpcodeModsWithOperandsMatcher,
)
from reprospect.test.sass.instruction.integer import (
    IntAdd3Matcher,
    IntAddMatcher,
)
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass.decode import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class AddInt32Matcher(InstructionMatcher):
    """
    Match the addition of two 32-bit integers.

    It can be achieved using one of the following instructions::

        IADD <dst>, <src_a>, <src_b>
        IADD3 <dst>, <src_a>, <src_b>, RZ
        IMAD.IADD <dst>, <src_a>, 0x1, <src_b>

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers',)

    def __init__(self, *,
        arch: NVIDIAArch,
        dst: str = Register.REG,
        src_a: str = Register.REG,
        src_b: str = Register.REG,
        swap: bool = False,
    ) -> None:
        """
        :param swap: If :py:obj:`True`, allow sources to be swapped.
        """
        matchers: list[OpcodeModsWithOperandsMatcher] = []

        matchers.append(IntAddMatcher(dst=dst, src_a=src_a, src_b=src_b))
        if swap:
            matchers.append(IntAddMatcher(dst=dst, src_a=src_b, src_b=src_a))

        matchers.append(IntAdd3Matcher(arch=arch, dst=dst, src_a=src_a, src_b=src_b, src_c='RZ'))
        if swap:
            matchers.append(IntAdd3Matcher(arch=arch, dst=dst, src_a=src_b, src_b=src_a, src_c='RZ'))

        matchers.append(OpcodeModsWithOperandsMatcher(
            opcode='IMAD', modifiers=('IADD',),
            operands=(dst, src_a, '0x1', src_b),
        ))
        if swap:
            matchers.append(OpcodeModsWithOperandsMatcher(
                opcode='IMAD', modifiers=('IADD',),
                operands=(dst, src_b, '0x1', src_a),
            ))

        self.matchers: typing.Final[list[OpcodeModsWithOperandsMatcher]] = matchers

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        for matcher in self.matchers:
            if (matched := matcher.match(inst=inst)) is not None:
                return matched
        return None
