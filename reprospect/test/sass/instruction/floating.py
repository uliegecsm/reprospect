"""
Collection of 32- and 64-bit floating point instruction matchers.
"""

import typing

import regex

from reprospect.test.sass.instruction.constant import Constant
from reprospect.test.sass.instruction.immediate import Immediate
from reprospect.test.sass.instruction.instruction import (
    PatternMatcher,
    ZeroOrOne,
)
from reprospect.test.sass.instruction.operand import Operand
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.test.sass.instruction.register import Register


def floating_point_add_pattern(*, ftype: typing.Literal['F', 'D']) -> regex.Pattern[str]:
    """
    Helper for:

    * :py:class:`reprospect.test.sass.instruction.floating.Fp32AddMatcher`.
    * :py:class:`reprospect.test.sass.instruction.floating.Fp64AddMatcher`.
    """
    return regex.compile(PatternMatcher.build_pattern(
        opcode=f'{ftype}ADD',
        modifiers=(ZeroOrOne('FTZ'),),
        operands=(
            Register.dst(captured=False),
            Operand.mod(Register.REGZ, math=None, captured=False),
            PatternBuilder.any(
                Operand.mod(Register.REGZ, math=None, captured=False),
                Register.UREG,
                Constant.ADDRESS,
                Immediate.FLOATING,
            ),
        ),
        predicate=False,
    ))

class Fp32AddMatcher(PatternMatcher):
    """
    Matcher for 32-bit floating-point add (``FADD``) instructions.
    """
    PATTERN: typing.Final[regex.Pattern[str]] = floating_point_add_pattern(ftype='F')

    def __init__(self) -> None:
        super().__init__(pattern=self.PATTERN)

class Fp64AddMatcher(PatternMatcher):
    """
    Matcher for 64-bit floating-point add (``DADD``) instructions.
    """
    PATTERN: typing.Final[regex.Pattern[str]] = floating_point_add_pattern(ftype='D')

    def __init__(self) -> None:
        super().__init__(pattern=self.PATTERN)
