import sys
import typing

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

OPERAND_MODIFIER: typing.Final[str] = r'[\-!\|~]'
"""Match any modifier of :py:class:`OperandModifier`."""

class OperandModifier(StrEnum):
    """
    Allowed operand value modifier.

    References:

    * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L67
    """
    NOT = '!'
    NEG = '-'
    ABS = '|'
    INV = '~'
