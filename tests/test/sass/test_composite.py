import typing

import pytest

from reprospect.test.sass.composite import (
    Fluentizer,
    any_of,
    findall,
    findunique,
    instruction_is,
    instructions_are,
    instructions_contain,
    interleaved_instructions_are,
    unordered_instructions_are,
    unordered_interleaved_instructions_are,
)
from reprospect.test.sass.composite_impl import (
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
from reprospect.test.sass.instruction import (
    AnyMatcher,
    Fp32AddMatcher,
    Fp64AddMatcher,
    InstructionMatch,
    OpcodeModsMatcher,
    RegisterMatcher,
)
from reprospect.tools.sass.decode import RegisterType


class TestInstructionIs:
    """
    Tests for :py:func:`reprospect.test.sass.composite.instruction_is`.
    """
    def test(self) -> None:
        matcher = instruction_is(Fp32AddMatcher())
        assert isinstance(matcher, Fluentizer)

        assert (matched := matcher.match(inst='FADD R2, R2, R3')) is not None
        assert isinstance(matched, InstructionMatch)

    def test_times_1(self) -> None:
        matcher = instruction_is(Fp32AddMatcher()).times(1)
        assert isinstance(matcher, InSequenceAtMatcher)

    def test_times_2(self) -> None:
        matcher = instruction_is(Fp32AddMatcher()).times(2)
        assert isinstance(matcher, OrderedInSequenceMatcher)
        assert len(matcher.matchers) == 2 and all(isinstance(x, Fp32AddMatcher) for x in matcher.matchers)

    def test_one_or_more_times(self) -> None:
        matcher = instruction_is(Fp32AddMatcher()).one_or_more_times()
        assert isinstance(matcher, OneOrMoreInSequenceMatcher)

    def test_zero_or_more_time(self) -> None:
        matcher = instruction_is(Fp32AddMatcher()).zero_or_more_times()
        assert isinstance(matcher, ZeroOrMoreInSequenceMatcher)

    def test_with_modifier(self) -> None:
        """
        Test :py:meth:`reprospect.test.sass.composite.Fluentizer.with_modifier`.
        """
        INSTRUCTION: typing.Final[str] = 'IMAD.WIDE.U32 R2, R0, R7, c[0x0][0x180]'
        MATCHER: typing.Final[Fluentizer] = instruction_is(AnyMatcher())

        assert MATCHER.with_modifier('WIDE', index=0).match(INSTRUCTION) is not None
        assert MATCHER.with_modifier('WIDE', index=1).match(INSTRUCTION) is None
        assert MATCHER.with_modifier('WIDE', index=None).match(INSTRUCTION) is not None

        assert MATCHER.with_modifier('U32', index=0).match(INSTRUCTION) is None
        assert MATCHER.with_modifier('U32', index=1).match(INSTRUCTION) is not None
        assert MATCHER.with_modifier('U32', index=None).match(INSTRUCTION) is not None

        assert MATCHER.with_modifier('HELLO', index=0).match(INSTRUCTION) is None

    def test_with_operand(self) -> None:
        """
        Test :py:meth:`reprospect.test.sass.composite.Fluentizer.with_operand`.
        """
        matcher = instruction_is(Fp32AddMatcher()).with_operand(
            index=1,
            operand='R2',
        )
        assert isinstance(matcher, Fluentizer)
        assert matcher.match(inst='FADD R2, R2, R3') is not None
        assert matcher.match(inst='FADD R4, R4, RZ') is None

        matcher = instruction_is(Fp32AddMatcher()).with_operand(
            index=2,
            operand=RegisterMatcher(rtype=RegisterType.GPR, special=False),
        )
        assert isinstance(matcher, Fluentizer)
        assert matcher.match(inst='FADD R2, R2, R3') is not None
        assert matcher.match(inst='FADD R2, R2, RZ') is None

        matcher = instruction_is(AnyMatcher()).with_operand(
            index=-1,
            operand=RegisterMatcher(rtype=RegisterType.UGPR, special=True),
        )
        assert matcher.match(inst='UIADD3 UR5, UPT, UPT, -UR4, UR9, URZ') is not None

        matcher = instruction_is(AnyMatcher()).with_operand(operand='UPT')
        assert matcher.match(inst='UIADD3 UR5, UPT, UPT, -UR4, UR9, URZ') is not None

        matcher = instruction_is(AnyMatcher()).with_operand(operand='R42')
        assert matcher.match(inst='UIADD3 UR5, UPT, UPT, -UR4, UR9, URZ') is None

    def test_with_operand_composed(self) -> None:
        """
        Similar to :py:meth:`test_with_operands` but calls
        :py:meth:`reprospect.test.sass.composite.Fluentizer.with_operand`
        many times.
        """
        matcher = instruction_is(Fp32AddMatcher()).with_operand(
            index=2, operand='R3',
        ).with_operand(
            index=1,
            operand=RegisterMatcher(rtype=RegisterType.GPR, index=2),
        )
        assert isinstance(matcher, Fluentizer)
        assert matcher.match(inst='FADD R2, R2, R3') is not None
        assert matcher.match(inst='FADD R3, R2, R3') is not None
        assert matcher.match(inst='FADD R2, R2, RZ') is None

    def test_with_operands(self) -> None:
        """
        Test :py:meth:`reprospect.test.sass.composite.Fluentizer.with_operands`.
        """
        matcher = instruction_is(AnyMatcher()).with_operands(operands=(
            (-1, 'URZ'),
            ( 1, RegisterMatcher(rtype=RegisterType.UPRED, special=True)),
        ))
        assert isinstance(matcher, Fluentizer)
        assert matcher.match(inst='UIADD3 UR5, UPT, UPT, -UR4, UR9, URZ') is not None

class TestInstructionsAre:
    """
    Tests for :py:func:`reprospect.test.sass.composite.instructions_are`.
    """
    def test(self) -> None:
        matcher = instructions_are(Fp32AddMatcher(), Fp32AddMatcher())
        assert isinstance(matcher, OrderedInSequenceMatcher)
        assert len(matcher.matchers) == 2 and all(isinstance(x, Fp32AddMatcher) for x in matcher.matchers)

    def test_mix(self) -> None:
        matcher = instructions_are(
            Fp32AddMatcher(),
            instruction_is(OpcodeModsMatcher(opcode='NOP', operands=False)).zero_or_more_times(),
            Fp32AddMatcher(),
        )
        assert isinstance(matcher, OrderedInSequenceMatcher)
        assert len(matcher.matchers) == 3

class TestUnorderedInstructionsAre:
    """
    Tests for :py:func:`reprospect.test.sass.composite.unordered_instructions_are`.
    """
    def test(self) -> None:
        matcher = unordered_instructions_are(Fp32AddMatcher(), Fp64AddMatcher())
        assert isinstance(matcher, UnorderedInSequenceMatcher)
        assert len(matcher.matchers) == 2
        assert isinstance(matcher.matchers[0], Fp32AddMatcher)
        assert isinstance(matcher.matchers[1], Fp64AddMatcher)

class TestInstructionsContain:
    """
    Tests for :py:func:`reprospect.test.sass.composite.instructions_contain`.
    """
    def test(self) -> None:
        matcher = instructions_contain(Fp32AddMatcher())
        assert isinstance(matcher, InSequenceMatcher)

class TestAnyOf:
    """
    Tests for :py:func:`reprospect.test.sass.composite.any_of`.
    """
    def test(self) -> None:
        matcher = any_of(Fp32AddMatcher(), Fp32AddMatcher())
        assert isinstance(matcher, AnyOfMatcher)

class TestInterleavedInstructionsAre:
    """
    Tests for :py:func:`reprospect.test.sass.composite.interleaved_instructions_are`.
    """
    def test(self) -> None:
        matcher = interleaved_instructions_are(Fp32AddMatcher(), Fp64AddMatcher())
        assert isinstance(matcher, OrderedInterleavedInSequenceMatcher)

class TestUnorderedInterleavedInstructionsAre:
    """
    Tests for :py:func:`reprospect.test.sass.composite.unordered_interleaved_instructions_are`.
    """
    def test(self) -> None:
        matcher = unordered_interleaved_instructions_are(Fp32AddMatcher(), Fp64AddMatcher())
        assert isinstance(matcher, UnorderedInterleavedInSequenceMatcher)

class TestFindall:
    """
    Tests for :py:func:`reprospect.test.sass.composite.findall`.
    """
    def test(self) -> None:
        matched = findall(Fp32AddMatcher(), (
            'FADD R0, R1, R2',
            'NOP',
            'NOP',
            'DADD R0, R1, R2',
            'FADD R45, R46, R47',
        ))
        assert len(matched) == 2

class TestFindunique:
    """
    Tests for :py:func:`reprospect.test.sass.composite.findunique`.
    """
    def test(self) -> None:
        with pytest.raises(ValueError):
            findunique(Fp32AddMatcher(), (
                'FADD R0, R1, R2',
                'NOP',
                'NOP',
                'DADD R0, R1, R2',
                'FADD R45, R46, R47',
            ))
