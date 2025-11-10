"""
Combine matchers from :py:mod:`reprospect.test.sass` into sequence matchers.
"""

import abc
import itertools
import sys
import typing

from reprospect.test.sass  import InstructionMatcher, InstructionMatch
from reprospect.tools.sass import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class SequenceMatcher(abc.ABC):
    """
    Base class for matchers of a sequence of instructions.
    """
    @abc.abstractmethod
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch] | None:
        pass

    @abc.abstractmethod
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch]:
        pass

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
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch] | None:
        matched = self.matcher.matches(instructions[start])
        return [matched] if matched is not None else None

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'{self.matcher!r} did not match {instructions[start]!r}.')
        return matched

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
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch] | None:
        matches : list[InstructionMatch] = []

        for instruction in instructions[start:]:
            if (matched := self.matcher.matches(instruction)) is not None:
                matches.append(matched)
            else:
                break

        return matches or None

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'{self.matcher!r} did not match {instructions[start]!r}.')
        return matched

class ZeroOrMoreInSequenceMatcher(OneOrMoreInSequenceMatcher):
    """
    Match zero or more times.
    """
    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch] | None:
        return super().matches(instructions = instructions, start = start) or []

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch]:
        """
        It is always matching.
        """
        return self.matches(instructions = instructions, start = start) or []

class OrderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of matchers in the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers',)

    def __init__(self, matchers : typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self.matchers : tuple[SequenceMatcher | InstructionMatcher, ...] = tuple(matchers)

    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch] | None:
        matches : list[InstructionMatch] = []
        for matcher in self.matchers:
            if isinstance(matcher, InstructionMatcher) and (single := matcher.matches(inst = instructions[start + len(matches)])) is not None:
                matches.append(single)
            elif isinstance(matcher, SequenceMatcher) and (many := matcher.matches(instructions = instructions, start = start + len(matches))) is not None:
                matches.extend(many)
            else:
                return None
        return matches

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'{self.matchers!r} did not match {instructions[start:start + len(self.matchers)]!r}.')
        return matched

class UnorderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of matchers in some permutation of the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers',)

    def __init__(self, matchers : typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self.matchers : tuple[SequenceMatcher | InstructionMatcher, ...] = tuple(matchers)

    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch] | None:
        """
        Cycle through all permutations of :py:attr:`matchers` (breaks on match).

        .. note::

            The implementation can be further optimized because it currently re-matches for each new permutation.
        """
        for permutation in itertools.permutations(self.matchers):
            if (matched := OrderedInSequenceMatcher(matchers = permutation).matches(instructions = instructions, start = start)) is not None:
                return matched
        return None

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'No permutation of {self.matchers!r} did match {instructions[start:start + len(self.matchers)]!r}.')
        return matched

class InSequenceMatcher(SequenceMatcher):
    """
    Check that a sequence contains an element that matches exactly.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher', 'index',)

    def __init__(self, matcher : SequenceMatcher | InstructionMatcher) -> None:
        self.matcher : typing.Final[OrderedInSequenceMatcher] = OrderedInSequenceMatcher(matchers = [matcher])
        self.index : int | None = None

    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch] | None:
        for index in range(start, len(instructions)):
            if (matched := self.matcher.matches(instructions = instructions, start = index)) is not None:
                self.index = index
                return matched
        return None

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[InstructionMatch]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'{self.matcher.matchers[0]!r} did not match.')
        return matched
