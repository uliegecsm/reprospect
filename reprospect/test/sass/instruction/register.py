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
from reprospect.tools.sass.decode import RegisterType

MODIFIER_REUSE: typing.Final[str] = 'reuse'

@dataclasses.dataclass(frozen=True, slots=True)
class RegisterMatch:
    """
    If :py:attr:`index` is :py:obj:`None`, it is a special register (*e.g.*
    ``RZ`` if :py:attr:`rtype` is :py:const:`reprospect.tools.sass.decode.RegisterType.GPR`).
    """
    rtype: RegisterType
    index: int | None = None
    reuse: bool = False
    math: MathModifier | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> RegisterMatch:
        captured = bits.capturesdict()

        if len(rtype := captured['rtype']) != 1:
            raise ValueError(bits)

        index: int | None = None
        if (value := captured.get('index')) is not None:
            if len(value) == 1:
                index = int(value[0])

        reuse: bool = False
        if (value := captured.get('reuse')) is not None:
            if len(value) == 1:
                reuse = True

        math: MathModifier | None = None
        if (value := captured.get('modifier_math')) is not None:
            if len(value) == 1:
                math = MathModifier(value[0])

        return cls(
            rtype=RegisterType(rtype[0]),
            index=index,
            reuse=reuse,
            math=math,
        )

@attrs.define(frozen=True, slots=True, kw_only=True)
class RegisterMatcher:
    """
    Matcher for a register.
    """
    rtype: RegisterType | None = None
    special: bool | None = None
    index: int | None = None
    reuse: bool | None = None
    math: MathModifier | None = None

    pattern: regex.Pattern[str] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        if self.special is not None and self.index is not None:
            raise RuntimeError(self)

        object.__setattr__(self, 'pattern', regex.compile(self.build_pattern(
            math=self.math,
            rtype=self.rtype,
            special=self.special,
            index=self.index,
            reuse=self.reuse,
            captured=False,
            capture_math=True,
            capture_reg=True,
            capture_reuse=True,
        )))

    @classmethod
    def build_pattern(cls, *,
        rtype: RegisterType | None = None,
        special: bool | None = None,
        index: int | None = None,
        reuse: bool | None = None,
        math: MathModifier | None = None,
        captured: bool = True,
        capture_math: bool = False,
        capture_reg: bool = False,
        capture_reuse: bool = False,
    ) -> str:
        pattern = ''.join(filter(None, (
            cls.build_pattern_modifier_math(math=math, captured=capture_math),
            cls.build_pattern_reg(rtype=rtype, special=special, index=index, captured=capture_reg),
            cls.build_pattern_modifier_reuse(reuse=reuse, captured=capture_reuse),
        )))
        return PatternBuilder.group(pattern, group='operands') if captured else pattern

    @classmethod
    def build_pattern_modifier_math(cls, *, math: MathModifier | None = None, captured: bool = True) -> str | None:
        if math is not None:
            return PatternBuilder.group(math.value, group='modifier_math') if captured else math.value
        inner = PatternBuilder.group(MODIFIER_MATH, group='modifier_math') if captured else MODIFIER_MATH
        return PatternBuilder.zero_or_one(inner)

    @classmethod
    def build_pattern_reg(cls, *, rtype: RegisterType | None = None, special: bool | None = None, index: int | None = None, captured: bool = True) -> str:
        if rtype is not None:
            inner_rtype = rtype.value
            inner_special = rtype.special
        else:
            inner_rtype = PatternBuilder.any('R', 'UR', 'P', 'UP')
            inner_special = PatternBuilder.any('Z', 'T')
        pattern_rtype = PatternBuilder.group(inner_rtype, group='rtype') if captured else inner_rtype

        if index is not None:
            return pattern_rtype + (PatternBuilder.group(index, group='index') if captured else str(index))

        if special is True:
            return pattern_rtype + (PatternBuilder.group(inner_special, group='special') if captured else inner_special)

        if special is False:
            return pattern_rtype + (PatternBuilder.group(r'\d+', group='index') if captured else r'\d+')

        pattern_special = PatternBuilder.group(inner_special, group='special') if captured else inner_special
        pattern_index   = PatternBuilder.group(r'\d+', group='index') if captured else r'\d+'
        return pattern_rtype + PatternBuilder.any(pattern_special, pattern_index)

    @classmethod
    def build_pattern_modifier_reuse(cls, *, reuse: bool | None = None, captured: bool = True) -> str | None:
        if reuse is True:
            return rf"\.{PatternBuilder.group(MODIFIER_REUSE, group='reuse')}" if captured else rf"\.{MODIFIER_REUSE}"
        if reuse is False:
            return None
        inner = PatternBuilder.group(MODIFIER_REUSE, group='reuse') if captured else MODIFIER_REUSE
        return PatternBuilder.zero_or_one(rf'\.{inner}')

    def match(self, reg: str) -> RegisterMatch | None:
        if (matched := self.pattern.match(reg)) is not None:
            return RegisterMatch.parse(bits=matched)
        return None

    def __call__(self, reg: str) -> RegisterMatch | None:
        return self.match(reg=reg)
