from __future__ import annotations

import dataclasses
import typing

import attrs
import regex

from reprospect.test.sass.instruction.operand import (
    MODIFIER_MATH,
    MathModifier,
)
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.test.sass.instruction.register import Register


class Constant:
    """
    Constant memory patterns.
    """
    BANK: typing.Final[str] = r'0x[0-9]+'
    """Constant memory bank."""

    OFFSET: typing.Final[str] = PatternBuilder.any(PatternBuilder.HEX, Register.REGZ, Register.UREG)
    """Constant memory offset."""

    ADDRESS: typing.Final[str] = r'c\[' + BANK + r'\]\[' + OFFSET + r'\]'
    """
    Constant memory location.
    """

    @classmethod
    def address(cls) -> str:
        return PatternBuilder.group(cls.ADDRESS, group='operands')

@dataclasses.dataclass(frozen=True, slots=True)
class ConstantMatch:
    """
    Result of matching a constant memory location.
    """
    bank: str
    offset: str
    math: MathModifier | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> ConstantMatch:
        captured = bits.capturesdict()

        if not (value := captured.get('bank')):
            raise ValueError(bits)
        bank = value[0]

        if not (value := captured.get('offset')):
            raise ValueError(bits)
        offset = value[0]

        math: MathModifier | None = None
        if (value := captured.get('modifier_math')) is not None:
            if len(value) == 1:
                math = MathModifier(value[0])

        return cls(bank=bank, offset=offset, math=math)

TEMPLATE_CONSTANT: typing.Final[str] = r'{modifier_math}c\[{bank}\]\[{offset}\]'

@attrs.define(frozen=True, slots=True, kw_only=True)
class ConstantMatcher:
    """
    Matcher for a constant memory location.
    """
    bank: str | None = None
    offset: str | None = None
    math: MathModifier | None = None

    pattern: regex.Pattern[str] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, 'pattern', regex.compile(self.build_pattern(
            bank=self.bank, offset=self.offset, math=self.math,
            captured=False, capture_bank=True, capture_offset=True, capture_modifier_math=True,
        )))

    @classmethod
    def build_pattern(cls, *,
        bank: str | None = None,
        offset: str | None = None,
        math: MathModifier | None = None,
        captured: bool = True,
        capture_bank: bool = False,
        capture_offset: bool = False,
        capture_modifier_math: bool = False,
    ) -> str:
        if math is None:
            pattern_modifier_math = PatternBuilder.zero_or_one(PatternBuilder.group(MODIFIER_MATH, group='modifier_math') if capture_modifier_math else MODIFIER_MATH)
        else:
            pattern_modifier_math = PatternBuilder.group(math.value, group='modifier_math') if capture_modifier_math else math.value

        pattern_bank   = bank   or Constant.BANK
        pattern_offset = offset or Constant.OFFSET

        pattern = TEMPLATE_CONSTANT.format(
            modifier_math=pattern_modifier_math,
            bank=PatternBuilder.group(  pattern_bank,   group='bank')   if capture_bank   else pattern_bank,
            offset=PatternBuilder.group(pattern_offset, group='offset') if capture_offset else pattern_offset,
        )

        return PatternBuilder.group(pattern, group='operands') if captured else pattern

    def match(self, constant: str) -> ConstantMatch | None:
        if (matched := self.pattern.match(constant)) is not None:
            return ConstantMatch.parse(bits=matched)
        return None

    def __call__(self, constant: str) -> ConstantMatch | None:
        return self.match(constant=constant)
