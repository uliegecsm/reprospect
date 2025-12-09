import typing

import pytest

from reprospect.test.sass.instruction import BranchMatcher, InstructionMatch

class TestBranchMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.BranchMatcher`.
    """
    INSTRUCTIONS : typing.Final[dict[str, InstructionMatch]] = {
        'BRA 0x240' : InstructionMatch(opcode = 'BRA', modifiers = (), operands = ('0x240',)),
        '@PT BRA 0xf0' : InstructionMatch(predicate = '@PT', opcode = 'BRA', modifiers = (), operands = ('0xf0',)),
        '@UPT BRA 0x170' : InstructionMatch(predicate = '@UPT', opcode = 'BRA', modifiers = (), operands = ('0x170',)),
        '@!UP0 BRA 0xc' : InstructionMatch(predicate = '@!UP0', opcode = 'BRA', modifiers = (), operands = ('0xc',)),
    }
    """
    Zoo of real ``BRA`` instructions.
    """

    @pytest.mark.parametrize(('instruction', 'expected'), INSTRUCTIONS.items())
    def test(self, instruction : str, expected : InstructionMatch) -> None:
        matcher = BranchMatcher()

        matched = matcher.match(inst = instruction)

        assert matched is not None
        assert matched == expected

    def test_with_predicate(self) -> None:
        """
        Verify that the matcher only matches instructions containing the exact predicate.
        Instructions without the predicate or with a different predicate should not match.
        """
        matcher = BranchMatcher(predicate = '@!UP0')

        assert matcher.match('@!UP0 BRA 0xc') == self.INSTRUCTIONS['@!UP0 BRA 0xc']
        assert matcher.match('@!UP2 BRA 0xc') is None
        assert matcher.match(      'BRA 0xc') is None

    def test_no_match(self) -> None:
        assert BranchMatcher().match(inst = 'STG.E [R2], R4') is None
