import sys
import typing

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

MODIFIER_MATH: typing.Final[str] = r'[\-!\|~]'
"""Match any math modifier from :py:class:`MathModifier`."""

class MathModifier(StrEnum):
    """
    Math operand modifier.
    """
    NOT = '!'
    NEG = '-'
    INV = '~'
    ABS = '|'
