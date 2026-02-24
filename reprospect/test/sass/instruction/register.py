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

class Register:
    """
    Register patterns.
    """
    #: Match a general purpose register.
    REG: typing.Final[str] = r'R[0-9]+'

    #: Match a general purpose register or ``RZ``.
    REGZ: typing.Final[str] = r'R(?:Z|\d+)'

    #: Match a uniform general purpose register.
    UREG: typing.Final[str] = r'UR[0-9]+'

    #: Match a uniform general purpose register or ``URZ``.
    UREGZ: typing.Final[str] = r'UR(?:Z|\d+)'

    #: Match a predicate register.
    PRED: typing.Final[str] = r'P[0-9]+'

    #: Match a predicate register or ``PT``.
    PREDT: typing.Final[str] = r'P(?:T|\d+)'

    #: Match a uniform predicate register.
    UPRED: typing.Final[str] = r'UP[0-9]+'

    #: Match a uniform predicate register or ``UPT``.
    UPREDT: typing.Final[str] = r'UP(?:T|\d+)'

    @classmethod
    def reg(cls) -> str:
        return PatternBuilder.group(cls.REG, group='operands')

    @classmethod
    def regz(cls) -> str:
        return PatternBuilder.group(cls.REGZ, group='operands')

    @classmethod
    def ureg(cls) -> str:
        return PatternBuilder.group(cls.UREG, group='operands')

    @classmethod
    def uregz(cls) -> str:
        return PatternBuilder.group(cls.UREGZ, group='operands')

    @classmethod
    def pred(cls) -> str:
        return PatternBuilder.group(cls.PRED, group='operands')

    @classmethod
    def predt(cls) -> str:
        return PatternBuilder.group(cls.PREDT, group='operands')

    @classmethod
    def upred(cls) -> str:
        return PatternBuilder.group(cls.UPRED, group='operands')

    @classmethod
    def upredt(cls) -> str:
        return PatternBuilder.group(cls.UPREDT, group='operands')

    @classmethod
    def dst(cls, *, captured: bool = True) -> str:
        if captured is True:
            return PatternBuilder.groups(cls.REG, groups=('dst', 'operands'))
        return PatternBuilder.group(cls.REG, group='dst')

    @classmethod
    def mod(cls, reg: str, *, reuse: bool | None = None, captured: bool = True) -> str:
        """
        Wrap a register pattern with a reuse modifier.
        """
        if reuse is None:
            inner = reg + PatternBuilder.zero_or_one(rf'\.{MODIFIER_REUSE}')
        elif reuse is True:
            inner = reg + rf'\.{MODIFIER_REUSE}'
        else:
            inner = reg
        return PatternBuilder.group(inner, group='operands') if captured else inner

@dataclasses.dataclass(frozen=True, slots=True)
class RegisterMatch:
    """
    If :py:attr:`index` is set to a negative value, it is a special register (*e.g.*
    ``RZ`` if :py:attr:`rtype` is :py:const:`reprospect.tools.sass.decode.RegisterType.GPR`).
    """
    rtype: RegisterType
    index: int = -1
    reuse: bool = False
    math: MathModifier | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> RegisterMatch:
        captured = bits.capturesdict()

        if len(rtype := captured['rtype']) != 1:
            raise ValueError(bits)

        index: int = -1
        if (value := captured.get('index')) is not None and len(value) == 1:
            index = int(value[0])

        reuse: bool = False
        if (value := captured.get('reuse')) is not None and len(value) == 1:
            reuse = True

        math: MathModifier | None = None
        if (value := captured.get('modifier_math')) is not None and len(value) == 1:
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
    math: MathModifier | bool | None = None

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
        math: MathModifier | bool | None = None,
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
    def build_pattern_modifier_math(cls, *, math: MathModifier | bool | None = None, captured: bool = True) -> str | None:
        if isinstance(math, MathModifier):
            return PatternBuilder.group(math.value, group='modifier_math') if captured else math.value
        if math is None or math is True:
            inner = PatternBuilder.group(MODIFIER_MATH, group='modifier_math') if captured else MODIFIER_MATH
            return PatternBuilder.zero_or_one(inner) if math is None else inner
        return None

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
