import typing

import regex

from reprospect.test.sass.instruction.instruction import PatternMatcher
from reprospect.test.sass.instruction.pattern import PatternBuilder


class BranchMatcher(PatternMatcher):
    """
    Matcher for a ``BRA`` branch instruction.

    Typically::

        @!UP5 BRA 0x456
    """
    PATTERN: typing.Final[regex.Pattern[str]] = regex.compile(PatternMatcher.build_pattern(
        opcode='BRA',
        modifiers=(),
        operands=(PatternBuilder.HEX,),
        predicate=None,
    ))

    def __init__(self, offset: str | None = None, predicate: str | None = None) -> None:
        super().__init__(pattern=self.PATTERN
            if not (offset or predicate)
            else self.build_pattern(
                opcode='BRA',
                modifiers=(),
                operands=(offset or PatternBuilder.HEX,),
                predicate=predicate,
            ),
        )
