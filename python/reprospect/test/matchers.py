"""
Thin user-facing factories for the matchers in :py:mod:`reprospect.test.matchers_impl`.
"""

import sys
import typing

from reprospect.tools.sass import Instruction
from reprospect.test       import matchers_impl, sass

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Fluentizer(sass.InstructionMatcher):
    """
    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, matcher : sass.InstructionMatcher) -> None:
        self.matcher : typing.Final[sass.InstructionMatcher] = matcher

    def times(self, num : int) -> matchers_impl.SequenceMatcher:
        """
        Match :py:attr:`matcher` `num` times (consecutively).

        .. note::

            If `num` is 0, it tests for the absence of a match.
        """
        if num < 0:
            raise RuntimeError()
        if num == 0:
            raise NotImplementedError()
        if num == 1:
            return matchers_impl.InSequenceAtMatcher(matcher = self.matcher)
        return matchers_impl.OrderedInSequenceMatcher(matchers = (self.matcher,) * num)

    def one_or_more_times(self) -> matchers_impl.OneOrMoreInSequenceMatcher:
        """
        Match :py:attr:`matcher` one or more times (consecutively).
        """
        return matchers_impl.OneOrMoreInSequenceMatcher(matcher = self.matcher)

    def zero_or_more_times(self) -> matchers_impl.ZeroOrMoreInSequenceMatcher:
        """
        Match :py:attr:`matcher` zero or more times (consecutively).
        """
        return matchers_impl.ZeroOrMoreInSequenceMatcher(matcher = self.matcher)

    @override
    @typing.final
    def matches(self, inst : Instruction | str) -> typing.Optional[sass.InstructionMatch]:
        return self.matcher.matches(inst = inst)

def instruction_is(matcher : sass.InstructionMatcher) -> Fluentizer:
    """
    Match the current instruction with `matcher`.

    >>> from reprospect.test.matchers import instruction_is
    >>> from reprospect.test.sass     import FloatAddMatcher
    >>> instruction_is(FloatAddMatcher()).matches(inst = 'FADD R2, R2, R3')
    InstructionMatch(opcode='FADD', modifiers=(), operands=('R2', 'R2', 'R3'), predicate=None, additional={'dst': ['R2']})
    >>> instruction_is(FloatAddMatcher()).one_or_more_times().matches(instructions = ('FADD R2, R2, R3', 'FADD R4, R4, R5'))
    [InstructionMatch(opcode='FADD', modifiers=(), operands=('R2', 'R2', 'R3'), predicate=None, additional={'dst': ['R2']}), InstructionMatch(opcode='FADD', modifiers=(), operands=('R4', 'R4', 'R5'), predicate=None, additional={'dst': ['R4']})]
    """
    return Fluentizer(matcher)

def instructions_are(*matchers : sass.InstructionMatcher | matchers_impl.SequenceMatcher) -> matchers_impl.OrderedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers`.

    >>> from reprospect.test.matchers import instructions_are
    >>> from reprospect.test.sass     import OpcodeModsMatcher
    >>> instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     instruction_is(OpcodeModsMatcher(opcode = 'NOP', operands = False)).zero_or_more_times(),
    ... ).matches(instructions = ('YIELD', 'NOP', 'NOP'))
    [InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None)]
    """
    return matchers_impl.OrderedInSequenceMatcher(matchers = matchers)

def unordered_instructions_are(*matchers : sass.InstructionMatcher | matchers_impl.SequenceMatcher) -> matchers_impl.UnorderedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers` (unordered).

    >>> from reprospect.test.matchers import unordered_instructions_are
    >>> from reprospect.test.sass     import OpcodeModsMatcher
    >>> unordered_instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... ).matches(instructions = ('NOP', 'YIELD'))
    [InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None)]
    """
    return matchers_impl.UnorderedInSequenceMatcher(matchers = matchers)

def instructions_contain(matcher : sass.InstructionMatcher | matchers_impl.SequenceMatcher) -> matchers_impl.InSequenceMatcher:
    """
    Check that a sequence of instructions contains at least one instruction matching `matcher`.

    ..note::

        Stops on the first match.

    >>> from reprospect.test.matchers import instructions_are, instructions_contain
    >>> from reprospect.test.sass     import OpcodeModsMatcher
    >>> matcher = instructions_contain(instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'FADD', operands = True),
    ... ))
    >>> matcher.matches(instructions = ('NOP', 'NOP', 'YIELD', 'FADD R1, R1, R2'))
    [InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='FADD', modifiers=(), operands=('R1', 'R1', 'R2'), predicate=None, additional=None)]
    >>> matcher.index
    2
    """
    return matchers_impl.InSequenceMatcher(matcher)

def any_of(*matchers : sass.InstructionMatcher | matchers_impl.SequenceMatcher) -> matchers_impl.AnyOfMatcher:
    """
    Match a sequence of instructions against any of the `matchers`.

    .. note::

        Returns the first match.

    >>> from reprospect.test.matchers import any_of
    >>> from reprospect.test.sass     import OpcodeModsMatcher
    >>> matcher = any_of(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... )
    >>> matcher.matches(instructions = ('FADD R1, R1, R2',)) is None
    True
    >>> matcher.index is None
    True
    """
    return matchers_impl.AnyOfMatcher(*matchers)
