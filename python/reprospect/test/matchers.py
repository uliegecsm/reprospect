"""
Combine matchers from :py:mod:`reprospect.test.sass` into sequence matchers.
"""

import abc
import dataclasses
import itertools
import sys
import typing

import regex

from reprospect.test.sass  import Matcher
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
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match] | None:
        pass

    @abc.abstractmethod
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match]:
        pass

@dataclasses.dataclass(slots = True, frozen = True)
class InSequenceAtMatcher(SequenceMatcher):
    """
    Check that the element matches exactly, raise otherwise.
    """
    matcher : Matcher

    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match] | None:
        matched = self.matcher.matches(instructions[start])
        return [matched] if matched is not None else None

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'{self.matcher!r} did not match {instructions[start]!r}.')
        return matched

@dataclasses.dataclass(slots = True, frozen = True)
class ZeroOrMoreInSequenceMatcher(SequenceMatcher):
    """
    Match zero or more times.
    """
    matcher : Matcher

    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match] | None:
        matches: list[regex.Match] = []

        for instruction in instructions[start:]:
            if (matched := self.matcher.matches(instruction)) is not None:
                matches.append(matched)
            else:
                break

        return matches

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match]:
        """
        It is always matching.
        """
        return self.matches(instructions = instructions, start = start) or []

@dataclasses.dataclass(slots = True, frozen = True)
class OrderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of matchers in the order they are provided.
    """
    matchers : tuple[SequenceMatcher | Matcher, ...]

    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match] | None:
        matches : list[regex.Match] = []
        for matcher in self.matchers:
            if isinstance(matcher, Matcher) and (matched := matcher.matches(inst = instructions[start + len(matches)])) is not None:
                matches.append(matched)
            elif isinstance(matcher, SequenceMatcher) and (matched := matcher.matches(instructions = instructions, start = start + len(matches))) is not None:
                matches.extend(matched)
            else:
                return None
        return matches

    @override
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'{self.matchers!r} did not match {instructions[start:start + len(self.matchers)]!r}.')
        return matched

@dataclasses.dataclass(slots = True, frozen = True)
class UnorderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of matchers in some permutation of the order they are provided.
    """
    matchers : tuple[SequenceMatcher | Matcher, ...]

    @override
    def matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match] | None:
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
    def assert_matches(self, instructions : typing.Sequence[Instruction], start : int = 0) -> list[regex.Match]:
        matched = self.matches(instructions = instructions, start = start)
        if matched is None:
            raise RuntimeError(f'No permutation of {self.matchers!r} did match {instructions[start:start + len(self.matchers)]!r}.')
        return matched
