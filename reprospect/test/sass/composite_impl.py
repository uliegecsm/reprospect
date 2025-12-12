"""
Combine matchers from :py:mod:`reprospect.test.sass.instruction` into sequence matchers.
"""

from __future__ import annotations

import abc
import sys
import typing

import attrs
import mypy_extensions

from reprospect.test.sass.instruction import (
    InstructionMatcher, InstructionMatch,
    AddressMatcher, ConstantMatcher, RegisterMatcher,
)
from reprospect.tools.sass import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@mypy_extensions.mypyc_attr(allow_interpreted_subclasses = True)
class SequenceMatcher(abc.ABC):
    """
    Base class for matchers of a sequence of instructions.
    """
    @abc.abstractmethod
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        """
        .. note::

            The `instructions` may be consumed more than once, *e.g.* in :py:class:`reprospect.test.sass.composite_impl.UnorderedInSequenceMatcher`.
            Therefore, it must be a :py:class:`typing.Sequence`, not a :py:class:`typing.Iterable`.
        """

    @typing.final
    def assert_matches(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch]:
        """
        Derived matchers are allowed to provide a nice message by implementing :py:meth:`explain`.
        """
        if (matched := self.match(instructions = instructions)) is None:
            raise RuntimeError(self.explain(instructions = instructions))
        return matched

    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str: # pylint: disable=unused-argument
        return f'{self!r} did not match.'

class InSequenceAtMatcher(SequenceMatcher):
    """
    Check that the element matches exactly.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, matcher : InstructionMatcher) -> None:
        self.matcher : typing.Final[InstructionMatcher] = matcher

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matched = self.matcher.match(instructions[0])
        return [matched] if matched is not None else None

    @override
    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str:
        return f'{self.matcher!r} did not match {instructions[0]!r}.'

class OneOrMoreInSequenceMatcher(SequenceMatcher):
    """
    Match one or more times.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, matcher : InstructionMatcher) -> None:
        self.matcher : typing.Final[InstructionMatcher] = matcher

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matches : list[InstructionMatch] = []

        for instruction in instructions:
            if (matched := self.matcher.match(instruction)) is not None:
                matches.append(matched)
            else:
                break

        return matches or None

    @override
    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str:
        return f'{self.matcher!r} did not match {instructions[0]!r}.'

class ZeroOrMoreInSequenceMatcher(OneOrMoreInSequenceMatcher):
    """
    Match zero or more times.
    """
    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        return super().match(instructions = instructions) or []

    @override
    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str:
        raise RuntimeError('It always matches.')

class OrderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers',)

    def __init__(self, matchers : typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self.matchers : typing.Final[tuple[SequenceMatcher | InstructionMatcher, ...]] = tuple(matchers)

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matches : list[InstructionMatch] = []

        for matcher in self.matchers:
            if isinstance(matcher, InstructionMatcher) and (single := matcher.match(inst = instructions[len(matches)])) is not None:
                matches.append(single)
            elif isinstance(matcher, SequenceMatcher) and (many := matcher.match(instructions[len(matches)::])) is not None:
                matches.extend(many)
            else:
                return None
        return matches

    @override
    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str:
        return f'{self.matchers!r} did not match {instructions!r}.'

class UnorderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in some permutation of the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers',)

    def __init__(self, matchers : typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self.matchers : typing.Final[tuple[SequenceMatcher | InstructionMatcher, ...]] = tuple(matchers)

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        return self.search(instructions = instructions, offset = 0, matchers = self.matchers) or None

    @classmethod
    def search(cls, instructions : typing.Sequence[Instruction | str], offset : int, matchers : tuple[SequenceMatcher | InstructionMatcher, ...]) -> list[InstructionMatch] | None:
        """
        Backtracking problem.
        """
        if not matchers:
            return []

        for index, matcher in enumerate(matchers):
            if isinstance(matcher, InstructionMatcher) and (single := matcher.match(instructions[offset])) is not None:
                if (inner := cls.search(instructions = instructions, offset = offset + 1, matchers = matchers[:index] + matchers[index + 1:])) is not None:
                    return [single] + inner
            elif isinstance(matcher, SequenceMatcher) and (many := matcher.match(instructions = instructions[offset:])) is not None:
                if (inner := cls.search(instructions = instructions, offset = offset + len(many), matchers = matchers[:index] + matchers[index + 1:])) is not None:
                    return many + inner
        return None

    @override
    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str:
        return f'No permutation of {self.matchers!r} did match {instructions!r}.'

class InSequenceMatcher(SequenceMatcher):
    """
    Check that a sequence contains an element that matches exactly.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher', 'index')

    def __init__(self, matcher : SequenceMatcher | InstructionMatcher) -> None:
        self.matcher : typing.Final[OrderedInSequenceMatcher] = OrderedInSequenceMatcher(matchers = (matcher,))
        self.index : int = -1

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        for index in range(len(instructions)):
            if (matched := self.matcher.match(instructions = instructions[index::])) is not None:
                self.index = index
                return matched
        return None

    @override
    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str:
        return f'{self.matcher.matchers[0]!r} did not match.'

class AnyOfMatcher(SequenceMatcher):
    """
    Match any of the :py:attr:`matchers`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers', 'index')

    def __init__(self, *matchers : SequenceMatcher | InstructionMatcher) -> None:
        self.matchers : typing.Final[tuple[SequenceMatcher | InstructionMatcher, ...]] = tuple(matchers)
        self.index : int = -1

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        """
        Loop over the :py:attr:`matchers` and return the first match.
        """
        for index, matcher in enumerate(self.matchers):
            if isinstance(matcher, InstructionMatcher) and (single := matcher.match(inst = instructions[0])) is not None:
                self.index = index
                return [single]
            if isinstance(matcher, SequenceMatcher) and (many := matcher.match(instructions = instructions)) is not None:
                self.index = index
                return many
        return None

    @override
    def explain(self, *, instructions : typing.Sequence[Instruction | str]) -> str:
        return f'None of {self.matchers!r} did match {instructions}.'

class OrderedInterleavedInSequenceMatcher(OrderedInSequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in the order they are provided, allowing interleaved unmatched instructions in between.
    """
    def __init__(self, matchers : typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        super().__init__(matchers = (InSequenceMatcher(matcher) for matcher in matchers))

class UnorderedInterleavedInSequenceMatcher(UnorderedInSequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in any order, allowing interleaved unmatched instructions in between.
    """
    def __init__(self, matchers: typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        super().__init__(matchers = (InSequenceMatcher(matcher) for matcher in matchers))

@attrs.define(frozen=True, slots=True)
class AllInSequenceMatcher:
    """
    Use :py:class:`InSequenceMatcher` to find all matches for :py:attr:`matcher` in a sequence of instructions.
    """
    matcher : InSequenceMatcher = attrs.field(converter=lambda x: x if isinstance(x, InSequenceMatcher) else InSequenceMatcher(x))

    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | list[list[InstructionMatch]]:
        if isinstance(self.matcher.matcher.matchers[0], InstructionMatcher):
            return self._match_single(instructions=instructions)
        return self._match_sequence(instructions=instructions)

    def _match_single(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch]:
        matches : list[InstructionMatch] = []
        offset = 0
        while (matched := self.matcher.match(instructions=instructions[offset:])):
            matches.append(matched[0])
            offset += self.matcher.index + 1
        return matches

    def _match_sequence(self, instructions : typing.Sequence[Instruction | str]) -> list[list[InstructionMatch]]:
        matches : list[list[InstructionMatch]] = []
        offset = 0
        while (matched := self.matcher.match(instructions=instructions[offset:])):
            matches.append(matched)
            offset += self.matcher.index + len(matched)
        return matches

class ModifierValidator(InstructionMatcher):
    """
    If :py:attr:`index` is an integer, the modifier must be present at that specific position in the matched instruction modifiers.
    If :py:attr:`index` is :py:obj:`None`, the modifier may be at any position in the matched instruction modifiers.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher', 'index', 'modifier')

    def __init__(self, matcher : InstructionMatcher, modifier : str, index : int | None = None) -> None:
        self.matcher : typing.Final[InstructionMatcher] = matcher
        self.index : typing.Final[int | None] = index
        self.modifier : typing.Final[str] = modifier

    @override
    def match(self, inst : Instruction | str) -> InstructionMatch | None:
        if (matched := self.matcher.match(inst)) is not None:
            if self.index is not None:
                try:
                    return matched if matched.modifiers[self.index] == self.modifier else None
                except IndexError:
                    return None
            else:
                return matched if self.modifier in matched.modifiers else None
        return None

OperandMatcher : typing.TypeAlias = str | AddressMatcher | ConstantMatcher | RegisterMatcher

class OperandValidator(InstructionMatcher):
    """
    Validate that the operand at :py:attr:`index` matches the instruction matched
    with :py:attr:`matcher`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher', 'index', 'operand')

    def __init__(self, matcher : InstructionMatcher, operand : OperandMatcher, index : int | None = None) -> None:
        self.matcher : typing.Final[InstructionMatcher] = matcher
        self.index : typing.Final[int | None] = index
        self.operand : typing.Final[OperandMatcher] = operand

    def check(self, operand : str) -> bool:
        if isinstance(self.operand, str) and operand == self.operand:
            return True
        return isinstance(self.operand, AddressMatcher | ConstantMatcher | RegisterMatcher) and self.operand.match(operand) is not None

    @override
    def match(self, inst : Instruction | str) -> InstructionMatch | None:
        if (matched := self.matcher.match(inst)) is not None:
            if self.index is not None:
                try:
                    return matched if self.check(operand = matched.operands[self.index]) else None
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

    def __init__(self, matcher : InstructionMatcher, operands : typing.Collection[tuple[int, OperandMatcher]]) -> None:
        self.matcher : typing.Final[InstructionMatcher] = matcher
        self.operands : typing.Final[tuple[tuple[int, OperandMatcher], ...]] = tuple(operands)

    @override
    def match(self, inst : Instruction | str) -> InstructionMatch | None:
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
