"""
Ensure that the distributed package supports user extensions.

``mypyc`` compilation can inadvertently seal classes, preventing inheritance.
This module verifies that :py:mod:`reprospect` extension points (*e.g.*,
:py:class:`reprospect.test.sass.instruction.InstructionMatcher`) remain
subclassable after compilation, allowing users to extend :py:mod:`reprospect` capabilities.
"""

import inspect
import logging
import sys
import typing

import pytest
import regex

from reprospect.tools.sass            import Instruction, ControlCode
from reprospect.test.sass.instruction import InstructionMatch, InstructionMatcher, PatternMatcher, Fp32AddMatcher
from reprospect.test.sass.composite   import instruction_is

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class NewInstructionMatcher(InstructionMatcher):
    """
    Faking a matcher for some unforeseen use case.
    """
    @override
    def match(self, inst: str | Instruction) -> InstructionMatch | None:
        if (matched := regex.match(r'(?P<opcode>[A-Z]+).*', inst.instruction if isinstance(inst, Instruction) else inst)) is not None:
            return InstructionMatch.parse(bits = matched)
        return None

class NewPatternMatcher(PatternMatcher):
    """
    Faking a matcher for some unforeseen use case.
    """
    def __init__(self):
        super().__init__(pattern = r'(?P<opcode>NOP|DADD|DMUL).*')

class CannotBeExtended(Fp32AddMatcher):
    """
    :py:class:`reprospect.test.sass.instruction.Fp32AddMatcher` was not marked as extensible.
    """

class TestInspect:
    """
    Use :py:mod:`inspect` to retrieve the provenance of objects.
    """
    def test_Instruction(self) -> None:
        logging.info(inspect.getfile(Instruction))

    def test_InstructionMatcher(self) -> None:
        logging.info(inspect.getfile(InstructionMatch))
        logging.info(inspect.getfile(InstructionMatcher))

class TestInstructionMatching:
    """
    Match instructions with :py:class:`NewInstructionMatcher` and :py:class:`NewPatternMatcher`.
    """
    CONTROLCODE: typing.Final[ControlCode] = ControlCode(
        stall_count = 5, yield_flag = False, read = 7, write = 7,
        wait = [True, False, False, False, False, False],
        reuse = {'A': False, 'B': False, 'C': False, 'D': False},
    )

    DADD: typing.Final[Instruction] = Instruction(offset = 0, instruction = 'DADD R4, R4, c[0x0][0x180]', hex = '0x0', control = CONTROLCODE)
    DMUL: typing.Final[Instruction] = Instruction(offset = 0, instruction = 'DMUL R6, R6, c[0x0][0x188]', hex = '0x1', control = CONTROLCODE)
    NOP: typing.Final[Instruction] = Instruction(offset = 0, instruction = 'NOP',                        hex = '0x2', control = CONTROLCODE)

    def test_NewInstructionMatcher(self) -> None:
        result = instruction_is(matcher = NewInstructionMatcher()).times(3).match(instructions = (self.DADD, self.DMUL, self.NOP))

        assert result is not None

        assert len(result) == 3

    def test_NewPatternMatcher(self) -> None:
        result = instruction_is(matcher = NewPatternMatcher()).times(3).match(instructions = (self.DADD, self.DMUL, self.NOP))

        assert result is not None

        assert len(result) == 3

class TestCannotBeExtended:
    def test(self) -> None:
        with pytest.raises(TypeError, match = 'interpreted classes cannot inherit from compiled'):
            CannotBeExtended()
