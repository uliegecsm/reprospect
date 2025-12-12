"""
Thin user-facing factories for the matchers in :py:mod:`reprospect.test.sass.composite_impl`.
"""
from __future__ import annotations

import sys
import typing

from reprospect.test.sass                import composite_impl, instruction
from reprospect.test.sass.composite_impl import OperandMatcher
from reprospect.tools.sass               import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Fluentizer(instruction.InstructionMatcher):
    """
    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('matcher',)

    def __init__(self, matcher : instruction.InstructionMatcher) -> None:
        self.matcher : typing.Final[instruction.InstructionMatcher] = matcher

    def times(self, num : int) -> composite_impl.InSequenceAtMatcher | composite_impl.OrderedInSequenceMatcher:
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
            return composite_impl.InSequenceAtMatcher(matcher = self.matcher)
        return composite_impl.OrderedInSequenceMatcher(matchers = (self.matcher,) * num)

    def one_or_more_times(self) -> composite_impl.OneOrMoreInSequenceMatcher:
        """
        Match :py:attr:`matcher` one or more times (consecutively).
        """
        return composite_impl.OneOrMoreInSequenceMatcher(matcher = self.matcher)

    def zero_or_more_times(self) -> composite_impl.ZeroOrMoreInSequenceMatcher:
        """
        Match :py:attr:`matcher` zero or more times (consecutively).
        """
        return composite_impl.ZeroOrMoreInSequenceMatcher(matcher = self.matcher)

    def with_modifier(self, modifier : str, index : int | None = None) -> Fluentizer:
        """
        >>> from reprospect.test.sass.composite import instruction_is
        >>> from reprospect.test.sass.instruction import AnyMatcher
        >>> matcher = instruction_is(AnyMatcher()).with_modifier(modifier='U32').with_operand(index=0, operand='R4')
        >>> matcher.match(inst='IMAD.WIDE.U32 R4, R7, 0x4, R4')
        InstructionMatch(opcode='IMAD', modifiers=('WIDE', 'U32'), operands=('R4', 'R7', '0x4', 'R4'), predicate=None, additional=None)
        """
        return Fluentizer(matcher = composite_impl.ModifierValidator(
            matcher = self.matcher,
            index = index, modifier = modifier,
        ))

    def with_operand(self, operand : OperandMatcher, index : int | None = None) -> Fluentizer:
        """
        >>> from reprospect.test.sass.composite   import instruction_is
        >>> from reprospect.test.sass.instruction import Fp32AddMatcher, RegisterMatcher
        >>> from reprospect.tools.sass.decode     import RegisterType
        >>> matcher = instruction_is(Fp32AddMatcher()).with_operand(index = 1, operand = RegisterMatcher(rtype = RegisterType.GPR, index = 8))
        >>> matcher.match(inst = 'FADD R5, R9, R10')
        >>> matcher.match(inst = 'FADD R5, R8, R9')
        InstructionMatch(opcode='FADD', modifiers=(), operands=('R5', 'R8', 'R9'), predicate=None, additional={'dst': ['R5']})
        """
        return Fluentizer(matcher = composite_impl.OperandValidator(
            matcher = self.matcher,
            index = index, operand = operand,
        ))

    def with_operands(self, operands : typing.Collection[tuple[int, OperandMatcher]]) -> Fluentizer:
        """
        Similar to :py:meth:`with_operand` for many operands.

        >>> from reprospect.test.sass.composite   import instruction_is
        >>> from reprospect.test.sass.instruction import Fp32AddMatcher
        >>> matcher = instruction_is(Fp32AddMatcher()).with_operands(
        ...     operands = ((1, 'R8'), (2, 'R9')),
        ... )
        >>> matcher.match(inst = 'FADD R5, R9, R10')
        >>> matcher.match(inst = 'FADD R5, R8, R9')
        InstructionMatch(opcode='FADD', modifiers=(), operands=('R5', 'R8', 'R9'), predicate=None, additional={'dst': ['R5']})
        """
        return Fluentizer(matcher = composite_impl.OperandsValidator(
            matcher = self,
            operands = operands,
        ))

    @override
    @typing.final
    def match(self, inst : Instruction | str) -> instruction.InstructionMatch | None:
        return self.matcher.match(inst = inst)

def instruction_is(matcher : instruction.InstructionMatcher) -> Fluentizer:
    """
    Match the current instruction with `matcher`.

    >>> from reprospect.test.sass.composite   import instruction_is
    >>> from reprospect.test.sass.instruction import Fp32AddMatcher
    >>> instruction_is(Fp32AddMatcher()).match(inst = 'FADD R2, R2, R3')
    InstructionMatch(opcode='FADD', modifiers=(), operands=('R2', 'R2', 'R3'), predicate=None, additional={'dst': ['R2']})
    >>> instruction_is(Fp32AddMatcher()).one_or_more_times().match(instructions = ('FADD R2, R2, R3', 'FADD R4, R4, R5'))
    [InstructionMatch(opcode='FADD', modifiers=(), operands=('R2', 'R2', 'R3'), predicate=None, additional={'dst': ['R2']}), InstructionMatch(opcode='FADD', modifiers=(), operands=('R4', 'R4', 'R5'), predicate=None, additional={'dst': ['R4']})]
    """
    return Fluentizer(matcher)

def instructions_are(*matchers : instruction.InstructionMatcher | composite_impl.SequenceMatcher) -> composite_impl.OrderedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers`.

    >>> from reprospect.test.sass.composite   import instructions_are
    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
    >>> instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     instruction_is(OpcodeModsMatcher(opcode = 'NOP', operands = False)).zero_or_more_times(),
    ... ).match(instructions = ('YIELD', 'NOP', 'NOP'))
    [InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None)]
    """
    return composite_impl.OrderedInSequenceMatcher(matchers = matchers)

def unordered_instructions_are(*matchers : instruction.InstructionMatcher | composite_impl.SequenceMatcher) -> composite_impl.UnorderedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers` (unordered).

    >>> from reprospect.test.sass.composite   import unordered_instructions_are
    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
    >>> unordered_instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... ).match(instructions = ('NOP', 'YIELD'))
    [InstructionMatch(opcode='NOP', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None)]
    """
    return composite_impl.UnorderedInSequenceMatcher(matchers = matchers)

def instructions_contain(matcher : instruction.InstructionMatcher | composite_impl.SequenceMatcher) -> composite_impl.InSequenceMatcher:
    """
    Check that a sequence of instructions contains at least one instruction matching `matcher`.

    .. note::

        Stops on the first match.

    >>> from reprospect.test.sass.composite   import instructions_are, instructions_contain
    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
    >>> matcher = instructions_contain(instructions_are(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'FADD', operands = True),
    ... ))
    >>> matcher.match(instructions = ('NOP', 'NOP', 'YIELD', 'FADD R1, R1, R2'))
    [InstructionMatch(opcode='YIELD', modifiers=(), operands=(), predicate=None, additional=None), InstructionMatch(opcode='FADD', modifiers=(), operands=('R1', 'R1', 'R2'), predicate=None, additional=None)]
    >>> matcher.index
    2
    """
    return composite_impl.InSequenceMatcher(matcher)

def any_of(*matchers : instruction.InstructionMatcher | composite_impl.SequenceMatcher) -> composite_impl.AnyOfMatcher:
    """
    Match a sequence of instructions against any of the `matchers`.

    .. note::

        Returns the first match.

    >>> from reprospect.test.sass.composite   import any_of
    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
    >>> matcher = any_of(
    ...     OpcodeModsMatcher(opcode = 'YIELD', operands = False),
    ...     OpcodeModsMatcher(opcode = 'NOP', operands = False),
    ... )
    >>> matcher.match(instructions = ('FADD R1, R1, R2',)) is None
    True
    >>> matcher.index is None
    True
    """
    return composite_impl.AnyOfMatcher(*matchers)

def interleaved_instructions_are(*matchers : instruction.InstructionMatcher | composite_impl.SequenceMatcher) -> composite_impl.OrderedInterleavedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers`, allowing matched instructions to be interleaved with unmatched instructions.

    >>> from reprospect.test.sass.composite import interleaved_instructions_are
    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
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
    return composite_impl.OrderedInterleavedInSequenceMatcher(matchers)

def unordered_interleaved_instructions_are(*matchers : instruction.InstructionMatcher | composite_impl.SequenceMatcher) -> composite_impl.UnorderedInterleavedInSequenceMatcher:
    """
    Match a sequence of instructions against `matchers` (unordered), allowing matched instructions to be interleaved with unmatched instructions.

    >>> from reprospect.test.sass.composite import unordered_interleaved_instructions_are
    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
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
    return composite_impl.UnorderedInterleavedInSequenceMatcher(matchers)
