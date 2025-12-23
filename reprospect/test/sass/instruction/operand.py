import sys
import typing

from reprospect.test.sass.instruction.pattern import PatternBuilder

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

MODIFIER_MATH: typing.Final[str] = r'[\-!\|~]'
"""Match any math modifier from :py:class:`MathModifier`."""

MODIFIER_MATH_ABS_DELIMITER: typing.Final[str] = r'\|'
"""Match :py:attr:`MathModifier.ABS`."""

class MathModifier(StrEnum):
    """
    Math operand modifier.
    """
    NOT = '!'
    NEG = '-'
    INV = '~'
    ABS = '|'

class Operand:
    """
    Operand patterns.
    """
    OPERAND: typing.Final[str] = r'[\w!\.\[\]\+\-\|~]+'
    """Match any operand."""

    @classmethod
    def operand(cls) -> str:
        return PatternBuilder.group(cls.OPERAND, group='operands')

    @classmethod
    def mod(cls, opnd: str, *, math: bool | None = None, captured: bool = True) -> str:
        """
        Wrap an operand pattern with a math modifier.
        """
        if math is None:
            inner = (
                PatternBuilder.zero_or_one(MODIFIER_MATH)
                + opnd
                + PatternBuilder.zero_or_one(MODIFIER_MATH_ABS_DELIMITER)
            )
        elif math is True:
            inner = (
                MODIFIER_MATH
                + opnd
                + PatternBuilder.zero_or_one(MODIFIER_MATH_ABS_DELIMITER)
            )
        else:
            inner = opnd
        return PatternBuilder.group(inner, group='operands') if captured else inner
