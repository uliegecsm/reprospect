import random

import pytest

from reprospect.test          import sass
from reprospect.test.matchers import InSequenceAtMatcher, \
                                     OrderedInSequenceMatcher, \
                                     UnorderedInSequenceMatcher, \
                                     ZeroOrMoreInSequenceMatcher
from reprospect.tools.sass    import ControlCode, Instruction

CONTROL_CODE = ControlCode.decode(code = '0x000e220000000800')

DADD = Instruction(offset = 0, instruction = 'DADD R4, R4, c[0x0][0x180]', hex = '0x0', control = CONTROL_CODE)
DMUL = Instruction(offset = 0, instruction = 'DMUL R6, R6, c[0x0][0x188]', hex = '0x1', control = CONTROL_CODE)
NOP  = Instruction(offset = 0, instruction = 'NOP',                        hex = '0x2', control = CONTROL_CODE)

DADD_DMUL = (DADD, DMUL)
"""One DADD followed by a DMUL."""

DADD_NOP_DMUL = (DADD, NOP, NOP, NOP, DMUL)
"""One DADD instruction followed by a few NOP, and a DMUL."""

NOP_DMUL_NOP_DADD = (NOP, DMUL, NOP, NOP, DADD)
"""NOP instructions with one DADD and one DMUL."""

MATCHER_DADD = sass.OpcodeModsWithOperandsMatcher(instruction = 'DADD', operands = ('R4', 'R4', sass.PatternBuilder.CONSTANT))
MATCHER_DMUL = sass.OpcodeModsWithOperandsMatcher(instruction = 'DMUL', operands = ('R6', 'R6', sass.PatternBuilder.CONSTANT))
MATCHER_NOP  = sass.OpcodeModsMatcher(instruction = 'NOP', operands = False)

class TestInSequenceAtMatcher:
    """
    Tests for :py:class:`reprospect.test.matchers.InSequenceAtMatcher`.
    """
    def test_match_first_element(self):
        """
        Matches the first element.
        """
        [matched] = InSequenceAtMatcher(matcher = MATCHER_DADD).matches(instructions = DADD_NOP_DMUL[0:1])

        assert matched.group(0) == DADD.instruction

    def test_match_with_start(self):
        """
        Matches the element pointed to by `start`.
        """
        [matched] = InSequenceAtMatcher(matcher = MATCHER_DADD).matches(instructions = NOP_DMUL_NOP_DADD, start = 4)

        assert matched.group(0) == DADD.instruction

    def test_no_match(self):
        """
        Raises if the first element does not match.
        """
        with pytest.raises(RuntimeError, match = 'did not match'):
            InSequenceAtMatcher(matcher = MATCHER_DADD).assert_matches(instructions = list(reversed(DADD_DMUL)))

class TestZeroOrMoreInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.matchers.ZeroOrMoreInSequenceMatcher`.
    """
    def test_matches_zero(self):
        """
        Matches zero time with sequence :py:const:`DADD_NOP_DMUL`.
        """
        matched = ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP).matches(instructions = DADD_NOP_DMUL)

        assert not matched

    def test_assert_matches_always_true(self):
        """
        :py:meth:`reprospect.test.matchers.ZeroOrMoreInSequenceMatcher.assert_matches` never raises.
        """
        assert not ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP).assert_matches(instructions = DADD_NOP_DMUL)

        assert len(ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP).assert_matches(instructions = NOP_DMUL_NOP_DADD)) == 1

    def test_matches_with_start(self):
        """
        Matches many times, with a `start` and sequence :py:const:`DADD_NOP_DMUL`.
        """
        matched = ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP).matches(instructions = DADD_NOP_DMUL, start = 1)

        assert len(matched) == 3 and all(x.group(0) == 'NOP' for x in matched)

class TestOrderedInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.matchers.OrderedInSequenceMatcher`.
    """
    def test_match_with_nop(self):
        """
        Matches the sequence :py:const:`DADD_NOP_DMUL`.
        """
        matched = OrderedInSequenceMatcher((
            MATCHER_DADD,
            ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
            MATCHER_DMUL,
        )).matches(instructions = DADD_NOP_DMUL)

        assert len(matched) == 5
        assert matched[0].group(0) == DADD.instruction
        assert all(x.group(0) == 'NOP' for x in matched[1:-2])
        assert matched[-1].group(0) == DMUL.instruction

    def test_match(self):
        """
        Matches the sequence :py:const:`DADD_DMUL`.
        """
        matched = OrderedInSequenceMatcher(
            matchers = (
                MATCHER_DADD,
                ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
                MATCHER_DMUL,
            ),
        ).matches(instructions = DADD_DMUL)

        assert len(matched) == 2
        assert matched[0].group(0) == DADD.instruction
        assert matched[1].group(0) == DMUL.instruction

    def test_no_match(self):
        """
        Does not match reversed sequence :py:const:`DADD_DMUL`.
        """
        with pytest.raises(RuntimeError, match = 'did not match'):
            OrderedInSequenceMatcher(
                matchers = (
                    MATCHER_DADD,
                    ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
                    MATCHER_DMUL,
                ),
            ).assert_matches(instructions = list(reversed(DADD_DMUL)))

class TestUnorderedInSequenceMatcher:
    """
    Tests for :py:class:`reprospect.test.matchers.UnorderedInSequenceMatcher`.
    """
    def test_match_with_nop(self):
        """
        Matches the sequence :py:const:`DADD_NOP_DMUL`.
        """
        inners = [
            ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        matched = UnorderedInSequenceMatcher(matchers = inners).matches(instructions = DADD_NOP_DMUL)

        assert len(matched) == 5

    def test_without_nop(self):
        """
        Matches the sequence :py:const:`DADD_DMUL`.
        """
        inners = [
            ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        matched = UnorderedInSequenceMatcher(matchers = inners).matches(instructions = DADD_DMUL)

        assert len(matched) == 2

    def test_with_split_nop(self):
        """
        Matches the sequence :py:const:`NOP_DMUL_NOP_DADD`.
        """
        inners = [
            ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
            ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        matched = UnorderedInSequenceMatcher(matchers = inners).matches(instructions = NOP_DMUL_NOP_DADD)

        assert len(matched) == 5

    def test_no_match(self):
        """
        All permutations fail on sequence :py:const:`NOP_DMUL_NOP_DADD`.
        """
        inners = [
            ZeroOrMoreInSequenceMatcher(matcher = MATCHER_NOP),
            MATCHER_DMUL,
            MATCHER_DADD,
        ]

        random.shuffle(inners)

        with pytest.raises(RuntimeError, match = 'No permutation of '):
            UnorderedInSequenceMatcher(matchers = inners).assert_matches(instructions = NOP_DMUL_NOP_DADD)
