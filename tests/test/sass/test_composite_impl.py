import random
import typing

import pytest

from reprospect.test.sass import instruction
from reprospect.test.sass.composite_impl import (
    AllInSequenceMatcher,
    AnyOfMatcher,
    InSequenceAtMatcher,
    InSequenceMatcher,
    OneOrMoreInSequenceMatcher,
    OrderedInSequenceMatcher,
    OrderedInterleavedInSequenceMatcher,
    UnorderedInSequenceMatcher,
    UnorderedInterleavedInSequenceMatcher,
    ZeroOrMoreInSequenceMatcher,
)
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass import ControlCode, Instruction

CONTROL_CODE = ControlCode.decode(code='0x000e220000000800')

DADD = Instruction(offset=0, instruction='DADD R4, R4, c[0x0][0x180]', hex='0x0', control=CONTROL_CODE)
DMUL = Instruction(offset=0, instruction='DMUL R6, R6, c[0x0][0x188]', hex='0x1', control=CONTROL_CODE)
NOP  = Instruction(offset=0, instruction='NOP',                        hex='0x2', control=CONTROL_CODE)

DADD_DMUL = (DADD, DMUL)
"""One DADD followed by a DMUL."""

DADD_NOP_DMUL = (DADD, NOP, NOP, NOP, DMUL)
"""One DADD instruction followed by a few NOP, and a DMUL."""

NOP_DMUL_NOP_DADD = (NOP, DMUL, NOP, NOP, DADD)
"""NOP instructions with one DADD and one DMUL."""

MATCHER_DADD = instruction.OpcodeModsWithOperandsMatcher(opcode='DADD', operands=('R4', 'R4', instruction.PatternBuilder.CONSTANT))
MATCHER_DMUL = instruction.OpcodeModsWithOperandsMatcher(opcode='DMUL', operands=('R6', 'R6', instruction.PatternBuilder.CONSTANT))
MATCHER_NOP  = instruction.OpcodeModsMatcher(opcode='NOP', operands=False)

class TestInSequenceAtMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.InSequenceAtMatcher`.
    """
    MATCHER: typing.Final[InSequenceAtMatcher] = InSequenceAtMatcher(matcher=MATCHER_DADD)

    def test_match_first_element(self) -> None:
        """
        Matches the first element.
        """
        matched = self.MATCHER.assert_matches(instructions=DADD_NOP_DMUL[0:1])
        assert len(matched) == 1
        assert matched[0].opcode == 'DADD'
        assert self.MATCHER.next_index == 1

    def test_match_with_start(self) -> None:
        """
        Matches the element pointed to by `start`.
        """
        matched = self.MATCHER.assert_matches(instructions=NOP_DMUL_NOP_DADD[4::])
        assert len(matched) == 1
        assert matched[0].opcode == 'DADD'
        assert self.MATCHER.next_index == 1

    def test_no_match(self) -> None:
        """
        Raises if the first element does not match.
        """
        with pytest.raises(RuntimeError, match='did not match'):
            self.MATCHER.assert_matches(instructions=list(reversed(DADD_DMUL)))

    def test_explain(self) -> None:
        assert self.MATCHER.explain(instructions=NOP_DMUL_NOP_DADD) == f'{MATCHER_DADD!r} did not match {NOP_DMUL_NOP_DADD[0]!r}.'

class TestZeroOrMoreInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.ZeroOrMoreInSequenceMatcher`.
    """
    def test_matches_zero(self) -> None:
        """
        Matches zero time with sequence :py:const:`DADD_NOP_DMUL`.
        """
        matcher = ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP)
        matched = matcher.match(instructions=DADD_NOP_DMUL)
        assert not matched
        assert matcher.next_index == 0

    def test_assert_matches_always_true(self) -> None:
        """
        Matching zero time is fine.
        """
        assert not ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP).assert_matches(instructions=DADD_NOP_DMUL)

        assert len(ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP).assert_matches(instructions=NOP_DMUL_NOP_DADD)) == 1

    def test_matches_with_start(self) -> None:
        """
        Matches many times, with a `start` and sequence :py:const:`DADD_NOP_DMUL`.
        """
        matcher = ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP)
        matched = matcher.assert_matches(instructions=DADD_NOP_DMUL[1::])
        assert len(matched) == 3 and all(x.opcode == 'NOP' for x in matched)
        assert matcher.next_index == 3

    def test_explain(self) -> None:
        with pytest.raises(RuntimeError, match='It always matches'):
            ZeroOrMoreInSequenceMatcher(matcher=MATCHER_DADD).explain(instructions=NOP_DMUL_NOP_DADD)

class TestOneOrMoreInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.OneOrMoreInSequenceMatcher`.
    """
    def test_matches_zero(self) -> None:
        """
        Does not match sequence :py:const:`DADD_NOP_DMUL`.
        """
        matcher = OneOrMoreInSequenceMatcher(matcher=MATCHER_NOP)

        assert not matcher.match(instructions=DADD_NOP_DMUL)

        with pytest.raises(RuntimeError, match='did not match'):
            matcher.assert_matches(instructions=DADD_NOP_DMUL)

    def test_matches_one(self) -> None:
        """
        Matches the sequence :py:const:`DADD_NOP_DMUL`.
        """
        matcher = OneOrMoreInSequenceMatcher(matcher=MATCHER_DADD)

        assert (matched := matcher.match(instructions=DADD_NOP_DMUL)) is not None
        assert len(matched) == 1
        assert matcher.next_index == 1

        assert (matched := matcher.assert_matches(instructions=DADD_NOP_DMUL)) is not None
        assert len(matched) == 1
        assert matcher.next_index == 1

    def test_matches_more(self) -> None:
        """
        Matches the sequence :py:const:`DADD_NOP_DMUL` many times starting at 1.
        """
        matcher = OneOrMoreInSequenceMatcher(matcher=MATCHER_NOP)

        assert (matched := matcher.match(instructions=DADD_NOP_DMUL[1::])) is not None
        assert len(matched) == 3
        assert matcher.next_index == 3

        assert (matched := matcher.assert_matches(instructions=DADD_NOP_DMUL[1::])) is not None
        assert len(matched) == 3
        assert matcher.next_index == 3

    def test_explain(self) -> None:
        assert OneOrMoreInSequenceMatcher(matcher=MATCHER_NOP).explain(instructions=DADD_NOP_DMUL) == f'{MATCHER_NOP} did not match {DADD_NOP_DMUL[0]}.'

class TestOrderedInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.OrderedInSequenceMatcher`.
    """
    MATCHER: typing.Final[OrderedInSequenceMatcher] = OrderedInSequenceMatcher(matchers=(
        MATCHER_DADD,
        ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP),
        MATCHER_DMUL,
    ))

    def test_match_with_nop(self) -> None:
        """
        Matches the sequence :py:const:`DADD_NOP_DMUL`.
        """
        assert (matched := self.MATCHER.match(instructions=DADD_NOP_DMUL)) is not None

        assert len(matched) == 5
        assert matched[0].opcode == 'DADD'
        assert all(x.opcode == 'NOP' for x in matched[1:-2])
        assert matched[-1].opcode == 'DMUL'
        assert self.MATCHER.next_index == 5

    def test_match(self) -> None:
        """
        Matches the sequence :py:const:`DADD_DMUL`.
        """
        assert (matched := self.MATCHER.match(instructions=DADD_DMUL)) is not None

        assert len(matched) == 2
        assert matched[0].opcode == 'DADD'
        assert matched[1].opcode == 'DMUL'
        assert self.MATCHER.next_index == 2

    def test_no_match(self) -> None:
        """
        Does not match reversed sequence :py:const:`DADD_DMUL`.
        """
        with pytest.raises(RuntimeError, match='did not match'):
            self.MATCHER.assert_matches(instructions=list(reversed(DADD_DMUL)))

    def test_explain(self) -> None:
        assert self.MATCHER.explain(instructions=DADD_NOP_DMUL) == f'{self.MATCHER.matchers!r} did not match {DADD_NOP_DMUL!r}.'

class TestUnorderedInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.UnorderedInSequenceMatcher`.
    """
    def test_match_with_nop(self) -> None:
        """
        Matches the sequence :py:const:`DADD_NOP_DMUL`.
        """
        inners: list[ZeroOrMoreInSequenceMatcher | instruction.OpcodeModsWithOperandsMatcher] = [
            ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        matcher = UnorderedInSequenceMatcher(matchers=inners)
        matched = matcher.assert_matches(instructions=DADD_NOP_DMUL)
        assert len(matched) == 5
        assert matcher.next_index == 5

    def test_without_nop(self) -> None:
        """
        Matches the sequence :py:const:`DADD_DMUL`.
        """
        inners: list[ZeroOrMoreInSequenceMatcher | instruction.OpcodeModsWithOperandsMatcher] = [
            ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        matcher = UnorderedInSequenceMatcher(matchers=inners)
        matched = matcher.assert_matches(instructions=DADD_DMUL)
        assert len(matched) == 2
        assert matcher.next_index == 2

    def test_with_split_nop(self) -> None:
        """
        Matches the sequence :py:const:`NOP_DMUL_NOP_DADD`.
        """
        inners: list[ZeroOrMoreInSequenceMatcher | instruction.OpcodeModsWithOperandsMatcher] = [
            ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP),
            ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        matcher = UnorderedInSequenceMatcher(matchers=inners)
        matched = matcher.assert_matches(instructions=NOP_DMUL_NOP_DADD)
        assert len(matched) == 5
        assert matcher.next_index == 5

    def test_no_match(self) -> None:
        """
        All permutations fail on sequence :py:const:`NOP_DMUL_NOP_DADD`.
        """
        inners: list[ZeroOrMoreInSequenceMatcher | instruction.OpcodeModsWithOperandsMatcher] = [
            ZeroOrMoreInSequenceMatcher(matcher=MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        with pytest.raises(RuntimeError, match='No permutation of '):
            UnorderedInSequenceMatcher(matchers=inners).assert_matches(instructions=NOP_DMUL_NOP_DADD)

    def test_explain(self) -> None:
        assert UnorderedInSequenceMatcher(matchers=(MATCHER_DADD,)).explain(instructions=NOP_DMUL_NOP_DADD) == f'No permutation of {(MATCHER_DADD,)} did match {NOP_DMUL_NOP_DADD}.'

class TestInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.InSequenceMatcher`.
    """
    def test_matches(self) -> None:
        matcher = InSequenceMatcher(matcher=MATCHER_DADD)

        assert matcher.match(instructions=NOP_DMUL_NOP_DADD) is not None
        assert matcher.next_index == 5

    def test_no_match(self) -> None:
        matcher = InSequenceMatcher(matcher=MATCHER_NOP)

        assert matcher.match(instructions=DADD_DMUL) is None

    def test_explain(self) -> None:
        assert InSequenceMatcher(matcher=MATCHER_NOP).explain(instructions=DADD_DMUL) == f'{MATCHER_NOP} did not match.'

class TestAnyOfMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.AnyOfMatcher`.
    """
    def test_matches(self) -> None:
        matcher = AnyOfMatcher(
            InSequenceMatcher(OrderedInSequenceMatcher(matchers=(MATCHER_NOP, MATCHER_NOP, MATCHER_DADD))),
            MATCHER_DADD,
        )

        assert matcher.match(instructions=NOP_DMUL_NOP_DADD) is not None
        assert matcher.matched == 0
        assert matcher.next_index == len(NOP_DMUL_NOP_DADD)

        assert matcher.match(instructions=(DADD,)) is not None

        assert matcher.matched == 1
        assert matcher.next_index == 1

    def test_no_match(self) -> None:
        matcher = AnyOfMatcher(
            MATCHER_DADD,
            MATCHER_DMUL,
        )

        assert matcher.match(instructions=(NOP, DADD, DMUL)) is None
        assert matcher.matched == -1

    def test_explain(self) -> None:
        assert AnyOfMatcher(MATCHER_DADD, MATCHER_DMUL).explain(instructions=DADD_NOP_DMUL) == f'None of {(MATCHER_DADD, MATCHER_DMUL)} did match {DADD_NOP_DMUL}.'

class TestOrderedInterleavedInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.OrderedInterleavedInSequenceMatcher`.
    """
    INSTRUCTIONS_DADD: typing.Final[tuple[str, ...]] = (
        'LDG.E.ENL2.256 R8, R4, desc[UR6][R2.64]',
        'DADD R4, R4, UR12',
        'NOP',
        'NOP',
        'NOP',
        'NOP',
        'DADD R6, R6, UR14',
        'NOP',
        'NOP',
        'NOP',
        'NOP',
        'DADD R8, R8, UR16',
        'NOP',
        'NOP',
        'NOP',
        'NOP',
        'DADD R10, R10, UR18',
        'STG.E.ENL2.256 desc[UR6][R2.64], R4, R8',
    )

    MATCHERS_DADD: typing.Final[tuple[instruction.OpcodeModsWithOperandsMatcher, ...]] = (
        instruction.OpcodeModsWithOperandsMatcher(opcode='DADD', modifiers=(), operands=('R4',  'R4',  instruction.PatternBuilder.UREG)),
        instruction.OpcodeModsWithOperandsMatcher(opcode='DADD', modifiers=(), operands=('R6',  'R6',  instruction.PatternBuilder.UREG)),
        instruction.OpcodeModsWithOperandsMatcher(opcode='DADD', modifiers=(), operands=('R8',  'R8',  instruction.PatternBuilder.UREG)),
        instruction.OpcodeModsWithOperandsMatcher(opcode='DADD', modifiers=(), operands=('R10', 'R10', instruction.PatternBuilder.UREG)),
    )

    INSTRUCTIONS_LDG: typing.Final[tuple[str, ...]] = (
        'LDG.E.U16.SYS R2, [R2]',
        'LDG.E.U16.SYS R4, [R4]',
    )

    MATCHERS_LDG: typing.Final[tuple[instruction.LoadGlobalMatcher, ...]] = (
        instruction.LoadGlobalMatcher(arch=NVIDIAArch.from_compute_capability(70), size=16, extend='U', readonly=False),
        instruction.LoadGlobalMatcher(arch=NVIDIAArch.from_compute_capability(70), size=16, extend='U', readonly=False),
    )

    def test_ldg(self) -> None:
        matcher = OrderedInterleavedInSequenceMatcher(self.MATCHERS_LDG)
        assert matcher.assert_matches(instructions=self.INSTRUCTIONS_LDG) == [
            instruction.InstructionMatch(opcode='LDG', modifiers=('E', 'U16', 'SYS'), operands=('R2', '[R2]'), additional={'address': ['[R2]']}),
            instruction.InstructionMatch(opcode='LDG', modifiers=('E', 'U16', 'SYS'), operands=('R4', '[R4]'), additional={'address': ['[R4]']}),
        ]
        assert matcher.next_index == 2

        assert matcher.match(instructions=self.INSTRUCTIONS_DADD) is None

    def test_dadd(self) -> None:
        matcher = OrderedInterleavedInSequenceMatcher(self.MATCHERS_DADD)
        assert matcher.assert_matches(instructions=self.INSTRUCTIONS_DADD) == [
            instruction.InstructionMatch(opcode='DADD', modifiers=(), operands=('R4', 'R4', 'UR12')),
            instruction.InstructionMatch(opcode='DADD', modifiers=(), operands=('R6', 'R6', 'UR14')),
            instruction.InstructionMatch(opcode='DADD', modifiers=(), operands=('R8', 'R8', 'UR16')),
            instruction.InstructionMatch(opcode='DADD', modifiers=(), operands=('R10', 'R10', 'UR18')),
        ]
        assert matcher.next_index == 17

        assert matcher.match(instructions=self.INSTRUCTIONS_DADD[::-1]) is None
        assert matcher.match(instructions=self.INSTRUCTIONS_LDG) is None

    def test_no_match(self) -> None:
        matcher = OrderedInterleavedInSequenceMatcher((
            instruction.OpcodeModsMatcher(opcode='YIELD', operands=False),
            instruction.OpcodeModsMatcher(opcode='NOP', operands=False),
        ))
        assert matcher.match(instructions=(
            'YIELD', 'YIELD',
        )) is None

class TestUnorderedInterleavedInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.UnorderedInterleavedInSequenceMatcher`.
    """
    def test_dadd(self) -> None:
        matchers = list(TestOrderedInterleavedInSequenceMatcher.MATCHERS_DADD)
        random.shuffle(matchers)

        matcher = UnorderedInterleavedInSequenceMatcher(matchers)
        matched = matcher.assert_matches(TestOrderedInterleavedInSequenceMatcher.INSTRUCTIONS_DADD)
        assert len(matched) == 4
        assert all(x.opcode == 'DADD' for x in matched)
        assert matcher.next_index == 17

class TestAllInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.composite_impl.AllInSequenceMatcher`.
    """
    def test_single(self) -> None:
        """
        The inner matcher is a :py:class:`reprospect.test.sass.instruction.InstructionMatcher`.
        """
        matcher = AllInSequenceMatcher(instruction.OpcodeModsMatcher(opcode='DADD'))
        assert isinstance(matcher.matcher, InSequenceMatcher)
        assert (matched := matcher.match(instructions=TestOrderedInterleavedInSequenceMatcher.INSTRUCTIONS_DADD))
        assert len(matched) == 4
        assert all(isinstance(x, instruction.InstructionMatch) for x in matched)

    def test_sequence(self) -> None:
        """
        The inner matcher is a :py:class:`reprospect.test.sass.composite_impl.SequenceMatcher`.
        """
        matcher = AllInSequenceMatcher(OrderedInSequenceMatcher(matchers=(
            MATCHER_NOP,
            MATCHER_NOP,
            MATCHER_NOP,
            MATCHER_NOP,
        )))
        assert isinstance(matcher.matcher, InSequenceMatcher)
        assert (matched := matcher.match(instructions=TestOrderedInterleavedInSequenceMatcher.INSTRUCTIONS_DADD))
        assert len(matched) == 3
        for x in matched:
            assert isinstance(x, list)
            assert len(x) == 4
            assert all(isinstance(y, instruction.InstructionMatch) for y in x)

    def test_no_match(self) -> None:
        matcher = AllInSequenceMatcher(instruction.OpcodeModsMatcher(opcode='FADD'))
        assert not matcher.match(TestOrderedInterleavedInSequenceMatcher.INSTRUCTIONS_DADD)
