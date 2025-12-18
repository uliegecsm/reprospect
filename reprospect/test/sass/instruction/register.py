from __future__ import annotations

import dataclasses
import typing

import attrs
import regex

from reprospect.test.sass.instruction.operand import OperandModifier, OPERAND_MODIFIER
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.tools.sass.decode import RegisterType

@dataclasses.dataclass(frozen = True, slots = True)
class RegisterMatch:
    """
    If :py:attr:`index` is :py:obj:`None`, it is a special register (*e.g.*
    ``RZ`` if :py:attr:`rtype` is :py:const:`reprospect.tools.sass.decode.RegisterType.GPR`).
    """
    rtype: RegisterType
    index: int | None = None
    reuse: bool = False
    modifier: OperandModifier | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> RegisterMatch:
        captured = bits.capturesdict()

        if len(rtype := captured['rtype']) != 1:
            raise ValueError(bits)

        index: int | None = None
        if (value := captured.get('index')) is not None:
            if len(value) == 1:
                index = int(value[0])

        modifier: OperandModifier | None = None
        if (value := captured.get('modifier')) is not None:
            if len(value) == 1:
                modifier = OperandModifier(value[0])

        return cls(
            rtype=RegisterType(rtype[0]),
            index=index,
            reuse=bool(captured.get('reuse')),
            modifier=modifier,
        )

REGISTER_MATCHER_MOD_PRE: typing.Final[str] = PatternBuilder.zero_or_one(PatternBuilder.group(OPERAND_MODIFIER, group='modifier'))

@attrs.define(frozen=True, slots=True, kw_only=True)
class RegisterMatcher:
    """
    Matcher for a register.
    """
    rtype: RegisterType | None = None
    special: bool | None = None
    index: int | None = None
    reuse: bool | None = None
    modifier: OperandModifier | None = None

    pattern: regex.Pattern[str] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        if self.special is not None and self.index is not None:
            raise RuntimeError(self)

        object.__setattr__(self, 'pattern', regex.compile(''.join(filter(None, (
            self._build_modifier(),
            PatternBuilder.group(self.rtype.value if self.rtype else 'R|UR|P|UP', group='rtype'),
            self._build_special(),
            self._build_index(),
            self._build_reuse(),
        )))))

    def match(self, reg: str) -> RegisterMatch | None:
        if (matched := self.pattern.match(reg)) is not None:
            return RegisterMatch.parse(bits=matched)
        return None

    def __call__(self, reg: str) -> RegisterMatch | None:
        return self.match(reg=reg)

    def _build_modifier(self) -> str | None:
        if self.modifier is None:
            return REGISTER_MATCHER_MOD_PRE
        return PatternBuilder.group(self.modifier.value, group='modifier')

    def _build_special(self) -> str | None:
        if self.index is not None:
            return None
        if self.special is False:
            return None

        idx = self.rtype.special if self.rtype is not None else r'Z|T'

        special = PatternBuilder.group(idx, group='special')

        if self.special is None:
            return special + '?'
        return special

    def _build_index(self) -> str | None:
        if self.special is True:
            return None
        if self.index is not None:
            return PatternBuilder.group(self.index, group='index')
        index = PatternBuilder.group(r'\d+', group='index')
        if self.special is None:
            return index + '?'
        return index

    def _build_reuse(self) -> str | None:
        if self.reuse is False:
            return None
        if self.special is True:
            return None
        if self.rtype is not None and self.rtype.is_predicate:
            return None
        reuse = PatternBuilder.group(r'\.reuse', group='reuse')
        if self.reuse is None:
            return reuse + '?'
        return reuse
