from reprospect.test.matchers      import any_of, Fluentizer, instructions_are, instructions_contain, instruction_is, unordered_instructions_are
from reprospect.test.matchers_impl import AnyOfMatcher, InSequenceMatcher, InSequenceAtMatcher, OneOrMoreInSequenceMatcher, OrderedInSequenceMatcher, UnorderedInSequenceMatcher, ZeroOrMoreInSequenceMatcher
from reprospect.test.sass          import FloatAddMatcher, InstructionMatch, OpcodeModsMatcher

class TestInstructionIs:
    """
    Tests for :py:func:`reprospect.test.matchers.instruction_is`.
    """
    def test(self) -> None:
        matcher = instruction_is(FloatAddMatcher())
        assert isinstance(matcher, Fluentizer)

        assert (matched := matcher.matches(inst = 'FADD R2, R2, R3')) is not None
        assert isinstance(matched, InstructionMatch)

    def test_times_1(self) -> None:
        matcher = instruction_is(FloatAddMatcher()).times(1)
        assert isinstance(matcher, InSequenceAtMatcher)

    def test_times_2(self) -> None:
        matcher = instruction_is(FloatAddMatcher()).times(2)
        assert isinstance(matcher, OrderedInSequenceMatcher)
        assert len(matcher.matchers) == 2 and all(isinstance(x, FloatAddMatcher) for x in matcher.matchers)

    def test_one_or_more_times(self) -> None:
        matcher = instruction_is(FloatAddMatcher()).one_or_more_times()
        assert isinstance(matcher, OneOrMoreInSequenceMatcher)

    def test_zero_or_more_time(self) -> None:
        matcher = instruction_is(FloatAddMatcher()).zero_or_more_times()
        assert isinstance(matcher, ZeroOrMoreInSequenceMatcher)

class TestInstructionsAre:
    """
    Tests for :py:func:`reprospect.test.matchers.instructions_are`.
    """
    def test(self) -> None:
        matcher = instructions_are(FloatAddMatcher(), FloatAddMatcher())
        assert isinstance(matcher, OrderedInSequenceMatcher)
        assert len(matcher.matchers) == 2 and all(isinstance(x, FloatAddMatcher) for x in matcher.matchers)

    def test_mix(self) -> None:
        matcher = instructions_are(
            FloatAddMatcher(),
            instruction_is(OpcodeModsMatcher(opcode = 'NOP', operands = False)).zero_or_more_times(),
            FloatAddMatcher()
        )
        assert isinstance(matcher, OrderedInSequenceMatcher)
        assert len(matcher.matchers) == 3

class TestUnorderedInstructionsAre:
    """
    Tests for :py:func:`reprospect.test.matchers.unordered_instructions_are`.
    """
    def test(self) -> None:
        matcher = unordered_instructions_are(FloatAddMatcher(), FloatAddMatcher())
        assert isinstance(matcher, UnorderedInSequenceMatcher)
        assert len(matcher.matchers) == 2 and all(isinstance(x, FloatAddMatcher) for x in matcher.matchers)

class TestInstructionsContain:
    """
    Tests for :py:func:`reprospect.test.matchers.instructions_contain`.
    """
    def test(self) -> None:
        matcher = instructions_contain(FloatAddMatcher())
        assert isinstance(matcher, InSequenceMatcher)

class TestAnyOf:
    """
    Tests for :py:func:`reprospect.test.matchers.any_of`.
    """
    def test(self) -> None:
        matcher = any_of(FloatAddMatcher(), FloatAddMatcher())
        assert isinstance(matcher, AnyOfMatcher)
