from __future__ import annotations

import sys
import typing

from reprospect.testing.binaries.sass.instruction.address import AddressMatcher
from reprospect.testing.binaries.sass.instruction.constant import ConstantMatcher
from reprospect.testing.binaries.sass.instruction.instruction import (
    InstructionMatch,
    InstructionMatcher,
)
from reprospect.testing.binaries.sass.instruction.register import RegisterMatcher
from reprospect.tools.binaries.sass.decoder import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class ModifierValidator(InstructionMatcher):
    """
    If :py:attr:`index` is an integer, the modifier must be present at that specific position in the matched instruction modifiers.
    If :py:attr:`index` is :py:obj:`None`, the modifier may be at any position in the matched instruction modifiers.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('index', 'matcher', 'modifier')

    def __init__(self, matcher: InstructionMatcher, modifier: str, index: int | None = None) -> None:
        self.matcher: typing.Final[InstructionMatcher] = matcher
        self.index: typing.Final[int | None] = index
        self.modifier: typing.Final[str] = modifier

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        if (matched := self.matcher.match(inst)) is not None:
            if self.index is not None:
                try:
                    return matched if matched.modifiers[self.index] == self.modifier else None
                except IndexError:
                    return None
            else:
                return matched if self.modifier in matched.modifiers else None
        return None

OperandMatcher: typing.TypeAlias = str | AddressMatcher | ConstantMatcher | RegisterMatcher

class OperandValidator(InstructionMatcher):
    """
    Validate that the operand at :py:attr:`index` matches the instruction matched
    with :py:attr:`matcher`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('index', 'matcher', 'operand')

    def __init__(self, matcher: InstructionMatcher, operand: OperandMatcher, index: int | None = None) -> None:
        self.matcher: typing.Final[InstructionMatcher] = matcher
        self.index: typing.Final[int | None] = index
        self.operand: typing.Final[OperandMatcher] = operand

    def check(self, operand: str) -> bool:
        if isinstance(self.operand, str) and operand == self.operand:
            return True
        return isinstance(self.operand, AddressMatcher | ConstantMatcher | RegisterMatcher) and self.operand.match(operand) is not None

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        if (matched := self.matcher.match(inst)) is not None:
            if self.index is not None:
                try:
                    return matched if self.check(operand=matched.operands[self.index]) else None
                except IndexError:
                    return None
            else:
                return matched if any(self.check(x) for x in matched.operands) else None
        return None

class OperandsValidator(InstructionMatcher):
    """
    Validate that the :py:attr:`operands` match the instruction matched
    with :py:attr:`matcher`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher', 'operands')

    def __init__(self, matcher: InstructionMatcher, operands: typing.Collection[tuple[int, OperandMatcher]]) -> None:
        self.matcher: typing.Final[InstructionMatcher] = matcher
        self.operands: typing.Final[tuple[tuple[int, OperandMatcher], ...]] = tuple(operands)

    @override
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        if (matched := self.matcher.match(inst)) is not None:
            for mindex, moperand in self.operands:
                try:
                    operand = matched.operands[mindex]
                except IndexError:
                    return None
                if isinstance(moperand, str) and operand != moperand:
                    return None
                if isinstance(moperand, AddressMatcher | ConstantMatcher | RegisterMatcher) and moperand.match(operand) is None:
                    return None
            return matched
        return None
