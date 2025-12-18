from __future__ import annotations

import dataclasses
import typing

import attrs
import regex

from reprospect.test.sass.instruction.operand import (
    OPERAND_MODIFIER_ABS,
    OPERAND_MODIFIER_MATH,
    OperandModifierMath,
)
from reprospect.test.sass.instruction.pattern import PatternBuilder


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantMatch:
    """
    Result of matching a constant memory location.
    """
    bank: str
    offset: str
    math: OperandModifierMath | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> ConstantMatch:
        captured = bits.capturesdict()

        if not (value := captured.get('bank')):
            raise ValueError(bits)
        bank = value[0]

        if not (value := captured.get('offset')):
            raise ValueError(bits)
        offset = value[0]

        math: OperandModifierMath | None = None
        if (value := captured.get('modifier_math')) is not None:
            if len(value) == 1:
                math = OperandModifierMath(value[0])

        return cls(bank=bank, offset=offset, math=math)

TEMPLATE_CONSTANT: typing.Final[str] = r'{modifier_math_pre}c\[{bank}\]\[{offset}\]{modifier_math_post}'

@attrs.define(frozen=True, slots=True, kw_only=True)
class ConstantMatcher:
    """
    Matcher for a constant memory location.
    """
    bank: str | None = None
    offset: str | None = None
    math: OperandModifierMath | None = None

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
        math: OperandModifierMath | None = None,
        captured: bool = True,
        capture_bank: bool = False,
        capture_offset: bool = False,
        capture_modifier_math: bool = False,
    ) -> str:
        if math is None:
            if capture_modifier_math:
                pattern_modifier_math_pre = PatternBuilder.zero_or_one(PatternBuilder.group(OPERAND_MODIFIER_MATH, group='modifier_math'))
            else:
                pattern_modifier_math_pre = PatternBuilder.zero_or_one(OPERAND_MODIFIER_MATH)
            pattern_modifier_math_post = PatternBuilder.zero_or_one(OPERAND_MODIFIER_ABS)
        else:
            if capture_modifier_math:
                pattern_modifier_math_pre = PatternBuilder.group(math.value, group='modifier_math')
            else:
                pattern_modifier_math_pre = math.value
            pattern_modifier_math_post = math.value if math == OperandModifierMath.ABS else ''

        pattern_bank   = bank   or PatternBuilder.CONSTANT_BANK
        pattern_offset = offset or PatternBuilder.CONSTANT_OFFSET

        pattern = TEMPLATE_CONSTANT.format(
            modifier_math_pre=pattern_modifier_math_pre,
            bank=PatternBuilder.group(  pattern_bank,   group='bank')   if capture_bank   else pattern_bank,
            offset=PatternBuilder.group(pattern_offset, group='offset') if capture_offset else pattern_offset,
            modifier_math_post=pattern_modifier_math_post,
        )

        return PatternBuilder.group(pattern, group='operands') if captured else pattern

    def match(self, constant: str) -> ConstantMatch | None:
        if (matched := self.pattern.match(constant)) is not None:
            return ConstantMatch.parse(bits=matched)
        return None

    def __call__(self, constant: str) -> ConstantMatch | None:
        return self.match(constant=constant)
