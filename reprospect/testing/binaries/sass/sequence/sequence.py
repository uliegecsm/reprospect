"""
Combine matchers from :py:mod:`reprospect.testing.binaries.sass.instruction` into sequence matchers.
"""

from __future__ import annotations

import abc
import sys
import typing

import attrs
import mypy_extensions

from reprospect.testing.binaries.sass.instruction import (
    InstructionMatch,
    InstructionMatcher,
    ModifierValidator,
    OperandMatcher,
    OperandsValidator,
    OperandValidator,
)
from reprospect.tools.binaries.sass.decode import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@mypy_extensions.mypyc_attr(allow_interpreted_subclasses=True)
class SequenceMatcher(abc.ABC):
    """
    Base class for matchers of a sequence of instructions.
    """
    @abc.abstractmethod
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        """
        .. note::

            The `instructions` may be consumed more than once, *e.g.* in :py:class:`reprospect.testing.binaries.sass.sequence.UnorderedInSequenceMatcher`.
            Therefore, it must be a :py:class:`typing.Sequence`, not a :py:class:`typing.Iterable`.
        """

    @property
    @abc.abstractmethod
    def next_index(self) -> int:
        """
        Return the next index in the sequence of instructions that can be matched.

        This is the index after the last matched instruction during the last call to :py:meth:`match`, *i.e.* how far
        this matcher consumed the sequence "plus one".

        The return value is only meaningful if the last call to :py:meth:`match` returned a non-:py:obj:`None` value.
        """

    @typing.final
    def assert_matches(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch]:
        """
        Derived matchers are allowed to provide a nice message by implementing :py:meth:`explain`.
        """
        if (matched := self.match(instructions=instructions)) is None:
            raise RuntimeError(self.explain(instructions=instructions))
        return matched

    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str: # pylint: disable=unused-argument
        return f'{self!r} did not match.'

class InSequenceAtMatcher(SequenceMatcher):
    """
    Check that the element matches exactly.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, matcher: InstructionMatcher) -> None:
        self.matcher: typing.Final[InstructionMatcher] = matcher

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matched = self.matcher.match(instructions[0])
        return [matched] if matched is not None else None

    @override
    @property
    def next_index(self) -> int:
        return 1

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        return f'{self.matcher!r} did not match {instructions[0]!r}.'

class OneOrMoreInSequenceMatcher(SequenceMatcher):
    """
    Match one or more times.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index', 'matcher')

    def __init__(self, matcher: InstructionMatcher) -> None:
        self._index: int = 0
        self.matcher: typing.Final[InstructionMatcher] = matcher

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matches: list[InstructionMatch] = []

        for instruction in instructions:
            if (matched := self.matcher.match(instruction)) is not None:
                matches.append(matched)
            else:
                break
        self._index = len(matches)
        return matches or None

    @override
    @property
    def next_index(self) -> int:
        return self._index

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        return f'{self.matcher!r} did not match {instructions[0]!r}.'

class ZeroOrMoreInSequenceMatcher(OneOrMoreInSequenceMatcher):
    """
    Match zero or more times.
    """
    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        return super().match(instructions=instructions) or []

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        raise RuntimeError('It always matches.')

class CountInSequenceMatcher(SequenceMatcher):
    """
    Count how many times it matches in a sequence.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index', 'count', 'matcher')

    def __init__(self, matcher: InstructionMatcher, count: int) -> None:
        self._index: int = 0
        self.count: typing.Final[int] = count
        self.matcher: typing.Final[InstructionMatcher] = matcher

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matches: list[InstructionMatch] = []

        for instruction in instructions:
            if (matched := self.matcher.match(instruction)) is not None:
                matches.append(matched)
                if len(matches) > self.count:
                    return None
        if len(matches) != self.count:
            return None
        self._index = len(instructions)
        return matches

    @override
    @property
    def next_index(self) -> int:
        return self._index

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        return f'{self.matcher!r} did not match {self.count} times in {instructions!r}.'

class OrderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index', 'matchers')

    def __init__(self, matchers: typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self._index: int = 0
        self.matchers: typing.Final[tuple[SequenceMatcher | InstructionMatcher, ...]] = tuple(matchers)

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matches: list[InstructionMatch] = []

        self._index = 0

        for matcher in self.matchers:
            if isinstance(matcher, InstructionMatcher) and (single := matcher.match(inst=instructions[self._index])) is not None:
                matches.append(single)
                self._index += 1
            elif isinstance(matcher, SequenceMatcher) and (many := matcher.match(instructions[self._index:])) is not None:
                matches.extend(many)
                self._index += matcher.next_index
            else:
                return None
        return matches

    @override
    @property
    def next_index(self) -> int:
        return self._index

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        return f'{self.matchers!r} did not match {instructions!r}.'

class UnorderedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in some permutation of the order they are provided.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index', 'matchers')

    def __init__(self, matchers: typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self._index: int = 0
        self.matchers: typing.Final[tuple[SequenceMatcher | InstructionMatcher, ...]] = tuple(matchers)

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        if (matched := self.search(instructions=instructions, offset=0, matchers=self.matchers)) is not None:
            self._index = matched[0]
            return matched[1]
        return None

    @classmethod
    def search(cls, *,
        instructions: typing.Sequence[Instruction | str],
        offset: int,
        matchers: tuple[SequenceMatcher | InstructionMatcher, ...],
    ) -> tuple[int, list[InstructionMatch]] | None:
        """
        Backtracking problem.
        """
        if not matchers:
            return offset, []

        for index, matcher in enumerate(matchers):
            if isinstance(matcher, InstructionMatcher) and (single := matcher.match(instructions[offset])) is not None:
                if (inner := cls.search(instructions=instructions, offset=offset + 1, matchers=matchers[:index] + matchers[index + 1:])) is not None:
                    return inner[0], [single] + inner[1]
            elif isinstance(matcher, SequenceMatcher) and (many := matcher.match(instructions=instructions[offset:])) is not None: # ruff:ignore[collapsible-if]
                if (inner := cls.search(instructions=instructions, offset=offset + matcher.next_index, matchers=matchers[:index] + matchers[index + 1:])) is not None:
                    return inner[0], many + inner[1]
        return None

    @override
    @property
    def next_index(self) -> int:
        return self._index

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        return f'No permutation of {self.matchers!r} did match {instructions!r}.'

class InSequenceMatcher(SequenceMatcher):
    """
    Check that a sequence contains an element that matches exactly.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index', 'matcher')

    def __init__(self, matcher: SequenceMatcher | InstructionMatcher) -> None:
        self._index: int = 0
        self.matcher: typing.Final[SequenceMatcher | InstructionMatcher] = matcher

    def _match_single(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        assert isinstance(self.matcher, InstructionMatcher)
        for index, instruction in enumerate(instructions):
            if (single := self.matcher.match(inst=instruction)) is not None:
                self._index = index + 1
                return [single]
        return None

    def _match_sequence(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        assert isinstance(self.matcher, SequenceMatcher)
        for index in range(len(instructions)):
            if (many := self.matcher.match(instructions=instructions[index:])) is not None:
                self._index = index + self.matcher.next_index
                return many
        return None

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        if isinstance(self.matcher, InstructionMatcher):
            return self._match_single(instructions=instructions)
        return self._match_sequence(instructions=instructions)

    @override
    @property
    def next_index(self) -> int:
        return self._index

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        return f'{self.matcher!r} did not match.'

class AnyOfMatcher(SequenceMatcher):
    """
    Match any of the :py:attr:`matchers`.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index', 'matched', 'matchers')

    def __init__(self, *matchers: SequenceMatcher | InstructionMatcher) -> None:
        self._index: int = 0
        self.matched: int = -1
        self.matchers: typing.Final[tuple[SequenceMatcher | InstructionMatcher, ...]] = tuple(matchers)

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        """
        Loop over the :py:attr:`matchers` and return the first match.
        """
        for index, matcher in enumerate(self.matchers):
            if isinstance(matcher, InstructionMatcher) and (single := matcher.match(inst=instructions[0])) is not None:
                self.matched = index
                self._index = 1
                return [single]
            if isinstance(matcher, SequenceMatcher) and (many := matcher.match(instructions=instructions)) is not None:
                self.matched = index
                self._index = matcher.next_index
                return many
        return None

    @override
    @property
    def next_index(self) -> int:
        return self._index

    @override
    def explain(self, *, instructions: typing.Sequence[Instruction | str]) -> str:
        return f'None of {self.matchers!r} did match {instructions}.'

class OrderedInterleavedInSequenceMatcher(SequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in the order they are provided, allowing interleaved unmatched instructions in between.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index', 'matchers')

    def __init__(self, matchers: typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        self._index: int = 0
        self.matchers: typing.Final[tuple[InSequenceMatcher, ...]] = tuple(
            matcher if isinstance(matcher, InSequenceMatcher) else InSequenceMatcher(matcher)
            for matcher in matchers
        )

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        matches: list[InstructionMatch] = []
        self._index = 0
        for matcher in self.matchers:
            if (many := matcher.match(instructions=instructions[self._index:])) is not None:
                matches.extend(many)
                self._index += matcher.next_index
            else:
                return None
        return matches or None

    @override
    @property
    def next_index(self) -> int:
        return self._index

class UnorderedInterleavedInSequenceMatcher(UnorderedInSequenceMatcher):
    """
    Match a sequence of :py:attr:`matchers` in any order, allowing interleaved unmatched instructions in between.
    """
    def __init__(self, matchers: typing.Iterable[SequenceMatcher | InstructionMatcher]) -> None:
        super().__init__(matchers=(
            InSequenceMatcher(matcher) if not isinstance(matcher, InSequenceMatcher) else matcher
            for matcher in matchers
        ))

@attrs.define(frozen=True, slots=True)
class AllInSequenceMatcher:
    """
    Use :py:class:`InSequenceMatcher` to find all matches for :py:attr:`matcher` in a sequence of instructions.
    """
    matcher: InSequenceMatcher = attrs.field(converter=lambda x: x if isinstance(x, InSequenceMatcher) else InSequenceMatcher(x))

    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | list[list[InstructionMatch]]:
        if isinstance(self.matcher.matcher, InstructionMatcher):
            return self._match_single(instructions=instructions)
        return self._match_sequence(instructions=instructions)

    def _match_single(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch]:
        matches: list[InstructionMatch] = []
        offset = 0
        while (matched := self.matcher.match(instructions=instructions[offset:])):
            matches.append(matched[0])
            offset += self.matcher.next_index
        return matches

    def _match_sequence(self, instructions: typing.Sequence[Instruction | str]) -> list[list[InstructionMatch]]:
        matches: list[list[InstructionMatch]] = []
        offset = 0
        while (matched := self.matcher.match(instructions=instructions[offset:])):
            matches.append(matched)
            offset += self.matcher.next_index
        return matches

class Fluentizer(InstructionMatcher):
    """
    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, matcher: InstructionMatcher) -> None:
        self.matcher: typing.Final[InstructionMatcher] = matcher

    def times(self, num: int) -> InSequenceAtMatcher | OrderedInSequenceMatcher:
        """
        Match :py:attr:`matcher` `num` times (consecutively).

        .. note::

            If `num` is 0, it tests for the absence of a match.
        """
        if num < 0:
            raise RuntimeError
        if num == 0:
            raise NotImplementedError
        if num == 1:
            return InSequenceAtMatcher(matcher=self.matcher)
        return OrderedInSequenceMatcher(matchers=(self.matcher,) * num)

    def one_or_more_times(self) -> OneOrMoreInSequenceMatcher:
        """
        Match :py:attr:`matcher` one or more times (consecutively).
        """
        return OneOrMoreInSequenceMatcher(matcher=self.matcher)

    def zero_or_more_times(self) -> ZeroOrMoreInSequenceMatcher:
        """
        Match :py:attr:`matcher` zero or more times (consecutively).
        """
        return ZeroOrMoreInSequenceMatcher(matcher=self.matcher)

    def with_modifier(self, modifier: str, index: int | None = None) -> Fluentizer:
        """
        >>> from reprospect.testing.binaries.sass.sequence import instruction_is
        >>> from reprospect.testing.binaries.sass.instruction import AnyMatcher
        >>> matcher = instruction_is(AnyMatcher()).with_modifier(modifier='U32').with_operand(index=0, operand='R4')
        >>> matcher.match(inst='IMAD.WIDE.U32 R4, R7, 0x4, R4')
        InstructionMatch(opcode='IMAD', modifiers=('WIDE', 'U32'), operands=('R4', 'R7', '0x4', 'R4'), predicate=None, additional=None)
        """
        return Fluentizer(matcher=ModifierValidator(
            matcher=self.matcher,
            index=index, modifier=modifier,
        ))

    def with_operand(self, operand: OperandMatcher, index: int | None = None) -> Fluentizer:
        """
        >>> from reprospect.testing.binaries.sass.sequence import instruction_is
        >>> from reprospect.testing.binaries.sass.instruction import Fp32AddMatcher, RegisterMatcher
        >>> from reprospect.tools.binaries.sass.decode     import RegisterType
        >>> matcher = instruction_is(Fp32AddMatcher()).with_operand(index = 1, operand = RegisterMatcher(rtype = RegisterType.GPR, index = 8))
        >>> matcher.match(inst = 'FADD R5, R9, R10')
        >>> matcher.match(inst = 'FADD R5, R8, R9')
        InstructionMatch(opcode='FADD', modifiers=(), operands=('R5', 'R8', 'R9'), predicate=None, additional={'dst': ['R5']})
        """
        return Fluentizer(matcher=OperandValidator(
            matcher=self.matcher,
            index=index, operand=operand,
        ))

    def with_operands(self, operands: typing.Collection[tuple[int, OperandMatcher]]) -> Fluentizer:
        """
        Similar to :py:meth:`with_operand` for many operands.

        >>> from reprospect.testing.binaries.sass.sequence   import instruction_is
        >>> from reprospect.testing.binaries.sass.instruction import Fp32AddMatcher
        >>> matcher = instruction_is(Fp32AddMatcher()).with_operands(
        ...     operands = ((1, 'R8'), (2, 'R9')),
        ... )
        >>> matcher.match(inst = 'FADD R5, R9, R10')
        >>> matcher.match(inst = 'FADD R5, R8, R9')
        InstructionMatch(opcode='FADD', modifiers=(), operands=('R5', 'R8', 'R9'), predicate=None, additional={'dst': ['R5']})
        """
        return Fluentizer(matcher=OperandsValidator(
            matcher=self,
            operands=operands,
        ))

    @override
    @typing.final
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        return self.matcher.match(inst=inst)

def instruction_is(matcher: InstructionMatcher) -> Fluentizer:
    """
    Match the current instruction with `matcher`.

    >>> from reprospect.testing.binaries.sass.sequence   import instruction_is
    >>> from reprospect.testing.binaries.sass.instruction import Fp32AddMatcher
    >>> instruction_is(Fp32AddMatcher()).match(inst = 'FADD R2, R2, R3')
    InstructionMatch(opcode='FADD', modifiers=(), operands=('R2', 'R2', 'R3'), predicate=None, additional={'dst': ['R2']})
    >>> instruction_is(Fp32AddMatcher()).one_or_more_times().match(instructions = ('FADD R2, R2, R3', 'FADD R4, R4, R5'))
    [InstructionMatch(opcode='FADD', modifiers=(), operands=('R2', 'R2', 'R3'), predicate=None, additional={'dst': ['R2']}), InstructionMatch(opcode='FADD', modifiers=(), operands=('R4', 'R4', 'R5'), predicate=None, additional={'dst': ['R4']})]
    """
    return Fluentizer(matcher)

def instruction_count_is(matcher: InstructionMatcher, count: int) -> CountInSequenceMatcher:
    """
    Match only if the `matcher` matches `count` times in the sequence.

    >>> from reprospect.testing.binaries.sass.sequence import instruction_count_is
    >>> from reprospect.testing.binaries.sass.instruction import Fp32AddMatcher
    >>> matcher = instruction_count_is(Fp32AddMatcher(), count=2)
    >>> matcher.match(instructions=('FADD R2, R2, R3',))
    >>> matcher.match(instructions=('FADD R2, R2, R3', 'FADD R4, R4, R5'))
    [InstructionMatch(opcode='FADD', modifiers=(), operands=('R2', 'R2', 'R3'), predicate=None, additional={'dst': ['R2']}), InstructionMatch(opcode='FADD', modifiers=(), operands=('R4', 'R4', 'R5'), predicate=None, additional={'dst': ['R4']})]
    """
    return CountInSequenceMatcher(matcher=matcher, count=count)

def instructions_are(*matchers: InstructionMatcher | SequenceMatcher) -> OrderedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers`.

    >>> from reprospect.testing.binaries.sass.sequence   import instructions_are
    >>> from reprospect.testing.binaries.sass.instruction import OpcodeModsMatcher
    >>> instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     instruction_is(OpcodeModsMatcher(opcode = 'NOP', operands = False)).zero_or_more_times(),
    ... ).match(instructions = ('YIELD', 'NOP', 'NOP'))
    [InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None)]
    """
    return OrderedInSequenceMatcher(matchers=matchers)

def unordered_instructions_are(*matchers: InstructionMatcher | SequenceMatcher) -> UnorderedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers` (unordered).

    >>> from reprospect.testing.binaries.sass.sequence   import unordered_instructions_are
    >>> from reprospect.testing.binaries.sass.instruction import OpcodeModsMatcher
    >>> unordered_instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... ).match(instructions = ('NOP', 'YIELD'))
    [InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None)]
    """
    return UnorderedInSequenceMatcher(matchers=matchers)

def instructions_contain(matcher: InstructionMatcher | SequenceMatcher) -> InSequenceMatcher:
    """
    Check that a sequence of instructions contains at least one instruction matching `matcher`.

    .. note::

        Stops on the first match.

    >>> from reprospect.testing.binaries.sass.sequence   import instructions_are, instructions_contain
    >>> from reprospect.testing.binaries.sass.instruction import OpcodeModsMatcher
    >>> matcher = instructions_contain(instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'FADD', operands = True),
    ... ))
    >>> matcher.match(instructions = ('NOP', 'NOP', 'YIELD', 'FADD R1, R1, R2'))
    [InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='FADD', modifiers=(), operands=('R1', 'R1', 'R2'), predicate=None, additional=None)]
    >>> matcher.next_index
    4
    """
    return InSequenceMatcher(matcher)

def any_of(*matchers: InstructionMatcher | SequenceMatcher) -> AnyOfMatcher:
    """
    Match a sequence of instructions against any of the `matchers`.

    .. note::

        Returns the first match.

    >>> from reprospect.testing.binaries.sass.sequence   import any_of
    >>> from reprospect.testing.binaries.sass.instruction import OpcodeModsMatcher
    >>> matcher = any_of(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... )
    >>> matcher.match(instructions = ('FADD R1, R1, R2',)) is None
    True
    >>> matcher.match(instructions=('NOP',)) is not None
    True
    >>> matcher.matched
    1
    """
    return AnyOfMatcher(*matchers)

def interleaved_instructions_are(*matchers: InstructionMatcher | SequenceMatcher) -> OrderedInterleavedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers`, allowing matched instructions to be interleaved with unmatched instructions.

    >>> from reprospect.testing.binaries.sass.sequence import interleaved_instructions_are
    >>> from reprospect.testing.binaries.sass.instruction import OpcodeModsMatcher
    >>> matcher = interleaved_instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... )
    >>> matcher.match(instructions = ('YIELD', 'NOP')) is not None
    True
    >>> matcher.match(instructions = ('NOP', 'YIELD')) is None
    True
    >>> matcher.match(instructions = ('YIELD', 'FADD R0, R1, R2', 'NOP')) is not None
    True
    """
    return OrderedInterleavedInSequenceMatcher(matchers)

def unordered_interleaved_instructions_are(*matchers: InstructionMatcher | SequenceMatcher) -> UnorderedInterleavedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers` (unordered), allowing matched instructions to be interleaved with unmatched instructions.

    >>> from reprospect.testing.binaries.sass.sequence import unordered_interleaved_instructions_are
    >>> from reprospect.testing.binaries.sass.instruction import OpcodeModsMatcher
    >>> matcher = unordered_interleaved_instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... )
    >>> matcher.match(instructions = ('YIELD', 'NOP')) is not None
    True
    >>> matcher.match(instructions = ('NOP', 'YIELD')) is not None
    True
    >>> matcher.match(instructions = ('YIELD', 'FADD R0, R1, R2', 'NOP')) is not None
    True
    >>> matcher.match(instructions = ('YIELD',)) is None
    True
    """
    return UnorderedInterleavedInSequenceMatcher(matchers)

@typing.overload
def findall(matcher: InstructionMatcher, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch]:
    ...

@typing.overload
def findall(matcher: SequenceMatcher, instructions: typing.Sequence[Instruction | str]) -> list[list[InstructionMatch]]:
    ...

def findall(
    matcher: InstructionMatcher | SequenceMatcher,
    instructions: typing.Sequence[Instruction | str],
) -> list[InstructionMatch] | list[list[InstructionMatch]]:
    """
    Find all matches for `matcher` in a sequence of instructions.
    Similarly to :py:func:`re.findall`, return an empty list if no match found.

    >>> from reprospect.testing.binaries.sass.sequence import findall
    >>> from reprospect.testing.binaries.sass.instruction import OpcodeModsMatcher
    >>> findall(
    ...     OpcodeModsMatcher(opcode='FADD', operands=True),
    ...     (
    ...         'NOP',
    ...         'FADD R1, R1, R2',
    ...         'NOP',
    ...         'FADD R3, R4, R5',
    ... ))
    [InstructionMatch(opcode='FADD', modifiers=(), operands=('R1', 'R1', 'R2'), predicate=None, additional=None), InstructionMatch(opcode='FADD', modifiers=(), operands=('R3', 'R4', 'R5'), predicate=None, additional=None)]
    """
    return AllInSequenceMatcher(matcher).match(instructions=instructions)

@typing.overload
def findunique(matcher: InstructionMatcher, instructions: typing.Sequence[Instruction | str]) -> InstructionMatch:
    ...

@typing.overload
def findunique(matcher: SequenceMatcher, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch]:
    ...

def findunique(matcher: InstructionMatcher | SequenceMatcher, instructions: typing.Sequence[Instruction | str]) -> InstructionMatch | list[InstructionMatch]:
    """
    Ensure that :py:meth:`findall` matches once.
    """
    matched = findall(matcher, instructions)
    if len(matched) != 1:
        raise ValueError
    return matched[0]
