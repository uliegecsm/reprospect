from __future__ import annotations

import dataclasses
import typing

import attrs
import regex

from reprospect.test.sass.instruction.operand import OperandModifier, OPERAND_MODIFIER
from reprospect.test.sass.instruction.pattern import PatternBuilder


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantMatch:
    """
    Result of matching a constant memory location.
    """
    bank: str
    offset: str
    modifier: OperandModifier | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> ConstantMatch:
        captured = bits.capturesdict()

        if not (value := captured.get('bank')):
            raise ValueError(bits)
        bank = value[0]

        if not (value := captured.get('offset')):
            raise ValueError(bits)
        offset = value[0]

        modifier: OperandModifier | None = None
        if (value := captured.get('modifier')) is not None:
            if len(value) == 1:
                modifier = OperandModifier(value[0])

        return cls(bank=bank, offset=offset, modifier=modifier)

TEMPLATE_CONSTANT: typing.Final[str] = r'{modifier}c\[{bank}\]\[{offset}\]'

CONSTANT_MATCHER_MOD_PRE: typing.Final[str] = PatternBuilder.zero_or_one(PatternBuilder.group(OPERAND_MODIFIER, group='modifier'))

@attrs.define(frozen=True, slots=True, kw_only=True)
class ConstantMatcher:
    """
    Matcher for a constant memory location.
    """
    bank: str | None = None
    offset: str | None = None
    modifier: OperandModifier | None = None

    pattern: regex.Pattern[str] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, 'pattern', regex.compile(self.build_pattern(
            bank=self.bank, offset=self.offset, modifier=self.modifier,
            captured = False, capture_bank = True, capture_offset = True,
        )))

    @classmethod
    def build_pattern(cls, *,
        bank: str | None = None,
        offset: str | None = None,
        modifier: OperandModifier | None = None,
        captured: bool = True,
        capture_bank: bool = False,
        capture_offset: bool = False,
    ) -> str:
        pattern_bank   = bank   or PatternBuilder.CONSTANT_BANK
        pattern_offset = offset or PatternBuilder.CONSTANT_OFFSET

        pattern = TEMPLATE_CONSTANT.format(
            modifier = PatternBuilder.group(modifier,       group='modifier') if modifier       else CONSTANT_MATCHER_MOD_PRE,
            bank     = PatternBuilder.group(pattern_bank,   group='bank')     if capture_bank   else pattern_bank,
            offset   = PatternBuilder.group(pattern_offset, group='offset')   if capture_offset else pattern_offset,
        )

        return PatternBuilder.group(pattern, group='operands') if captured else pattern

    def match(self, constant: str) -> ConstantMatch | None:
        if (matched := self.pattern.match(constant)) is not None:
            return ConstantMatch.parse(bits=matched)
        return None

    def __call__(self, constant: str) -> ConstantMatch | None:
        return self.match(constant=constant)
