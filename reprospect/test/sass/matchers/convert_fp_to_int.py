import sys
import typing

from reprospect.test.sass.instruction import (
    InstructionMatch,
    InstructionMatcher,
    OpcodeModsWithOperandsMatcher,
)
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.sass.decode import Instruction
from reprospect.utils.types import ConvertibleTypeInfo, TypeInfo

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class ConvertFpToInt(InstructionMatcher):
    """
    Convert floating-point value to integer value.

    .. note::

        The `.NTZ` opcode modifier means::

            round to nearest integer in the direction of zero

        References:

        * :cite:`abdelkhalik-demystifying-2022`
        * https://docs.nvidia.com/cuda/parallel-thread-execution/#rounding-modifiers

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, *,
        dst_dtype: ConvertibleTypeInfo | None = None,
        src_dtype: ConvertibleTypeInfo | None = None,
        dst: str = Register.REG,
        src: str = Register.REG,
    ) -> None:
        """
        :param src_dtype: Type of the source floating-point value.
        :param dst_dtype: Type of the destination integer value.
        """
        src_dtype = TypeInfo.normalize(dtype=src_dtype or 32)
        dst_dtype = TypeInfo.normalize(dtype=dst_dtype or 32)

        modifiers = []

        if dst_dtype.bits != 32 or not dst_dtype.signed:
            modifiers.append(f'{"S" if dst_dtype.signed else "U"}{dst_dtype.bits}')
        if src_dtype.bits == 64:
            modifiers.append('F64')

        modifiers.append('TRUNC')

        if dst_dtype.bits == 32 and src_dtype.bits == 32:
            modifiers.append('NTZ')

        self.matcher: typing.Final[OpcodeModsWithOperandsMatcher] = OpcodeModsWithOperandsMatcher(opcode='F2I', modifiers=modifiers, operands=(dst, src))

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        return self.matcher.match(inst=inst)
