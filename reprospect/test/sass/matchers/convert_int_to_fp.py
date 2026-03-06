import sys
import typing

from reprospect.test.sass.instruction import (
    InstructionMatch,
    InstructionMatcher,
    OpcodeModsWithOperandsMatcher,
)
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass.decode import Instruction
from reprospect.utils.types import ConvertibleTypeInfo, TypeInfo

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class ConvertIntToFp(InstructionMatcher):
    """
    Convert integer value to floating-point value.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, *,
        arch: NVIDIAArch,
        dst_dtype: ConvertibleTypeInfo | None = None,
        src_dtype: ConvertibleTypeInfo | None = None,
        dst: str = Register.REG,
        src: str = Register.REG,
    ) -> None:
        """
        :param src_type: Type of the source integer value.
        :param dst_type: Type of the destination floating-point value.
        """
        src_dtype = TypeInfo.normalize(dtype=src_dtype or 32)
        dst_dtype = TypeInfo.normalize(dtype=dst_dtype or 32)

        if arch.compute_capability >= 86 and dst_dtype.bits == 32 and src_dtype.bits == 32:
            opcode = 'I2FP'
            modifiers = ['F32', self.src_modifier(dtype=src_dtype)]
        else:
            opcode = 'I2F'
            modifiers = []
            if dst_dtype.bits == 64:
                modifiers.append('F64')
            if src_dtype.bits != 32 or not src_dtype.signed:
                modifiers.append(self.src_modifier(dtype=src_dtype))

        self.matcher: typing.Final[OpcodeModsWithOperandsMatcher] = OpcodeModsWithOperandsMatcher(opcode=opcode, modifiers=modifiers, operands=(dst, src))

    @staticmethod
    def src_modifier(*, dtype: TypeInfo) -> str:
        return f'{"S" if dtype.signed else "U"}{dtype.bits}'

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        return self.matcher.match(inst=inst)
