"""
Combine matchers from :py:mod:`reprospect.test.sass.instruction` into sequence matchers.
"""

import abc
import itertools
import sys
import typing

import mypy_extensions

from reprospect.test.sass.instruction import InstructionMatcher, InstructionMatch, RegisterMatcher
from reprospect.tools.sass            import Instruction

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
    Match a sequence of matchers in the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers',)

    def __init__(self, matchers : typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self.matchers : tuple[SequenceMatcher | InstructionMatcher, ...] = tuple(matchers)

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
    Match a sequence of matchers in some permutation of the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matchers',)

    def __init__(self, matchers : typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self.matchers : tuple[SequenceMatcher | InstructionMatcher, ...] = tuple(matchers)

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        """
        Cycle through all permutations of :py:attr:`matchers` (breaks on match).

        .. note::

            The implementation can be further optimized because it currently re-matches for each new permutation.
        """
        for permutation in itertools.permutations(self.matchers):
            if (matched := OrderedInSequenceMatcher(matchers = permutation).match(instructions = instructions)) is not None:
                return matched
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
        self.matcher : typing.Final[OrderedInSequenceMatcher] = OrderedInSequenceMatcher(matchers = [matcher])
        self.index : int | None = None

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
        self.index : int | None = None

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

OperandMatcher : typing.TypeAlias = str | RegisterMatcher

class OperandValidator(InstructionMatcher):
    """
    Validate that the operand at :py:attr:`index` matches the instruction matched
    with :py:attr:`matcher`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher', 'index', 'operand')

    def __init__(self, matcher : InstructionMatcher, index : int, operand : OperandMatcher) -> None:
        self.matcher : typing.Final[InstructionMatcher] = matcher
        self.index : typing.Final[int] = index
        self.operand : typing.Final[OperandMatcher] = operand

    @override
    def match(self, inst : Instruction | str) -> InstructionMatch | None:
        if (matched := self.matcher.match(inst)) is not None:
            try:
                operand = matched.operands[self.index]
            except IndexError:
                return None
            if isinstance(self.operand, str) and operand == self.operand:
                return matched
            if isinstance(self.operand, RegisterMatcher) and self.operand.match(operand) is not None:
                return matched
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
                if isinstance(moperand, RegisterMatcher) and moperand.match(operand) is None:
                    return None
            return matched
        return None
