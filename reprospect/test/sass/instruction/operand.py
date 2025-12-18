import sys
import typing

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

OPERAND_MODIFIER_MATH: typing.Final[str] = r'[\-!\|~]'
"""Match any math modifier (NOT, NEG, INV, ABS)."""

OPERAND_MODIFIER_ABS: typing.Final[str] = r'\|'
"""Match absolute value delimiter."""

class OperandModifierMath(StrEnum):
    """
    Math operand modifier.
    """
    NOT = '!'
    NEG = '-'
    INV = '~'
    ABS = '|'
