"""
Collection of integer instruction matchers.
"""

import typing

import regex

from reprospect.test.sass.instruction import (
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
    PatternMatcher,
)
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch


class IntAddMatcher(OpcodeModsWithOperandsMatcher):
    """
    Matcher for two 32-bit integer addition (``IADD``), such as::

        IADD R14, R8, R11
    """
    def __init__(self, *,
        src_a: str = Register.REG,
        src_b: str = Register.REG,
        dst: str = Register.REG,
    ) -> None:
        super().__init__(
            opcode='IADD',
            operands=(dst, src_a, src_b),
        )

class IntAdd3Matcher(OpcodeModsWithOperandsMatcher):
    """
    Matcher for three 32-bit integer addition (``IADD3``).

    ``IADD3`` is architecture-dependent.
    Prior to :py:attr:`reprospect.tools.architecture.NVIDIAFamily.BLACKWELL`, it is::

        IADD3 R7, R2, R7, RZ

    and as of :py:attr:`reprospect.tools.architecture.NVIDIAFamily.BLACKWELL`::

        IADD3 R17, PT, PT, R14, R11, R8
    """
    def __init__(self, *,
        arch: NVIDIAArch,
        src_a: str = Register.REG,
        src_b: str = Register.REG,
        src_c: str = Register.REGZ,
        dst: str = Register.REG,
    ) -> None:
        super().__init__(
            opcode='IADD3',
            operands=(
                dst,
                *(('PT', 'PT') if arch.compute_capability.as_int >= 100 else ()),
                src_a, src_b, src_c,
            ),
        )

class LEAMatcher(PatternMatcher):
    """
    Matcher for ``LEA`` instruction, such as::

        LEA R24, R4, R10, 0x3

    .. note::

        ``LEA`` typically computes an integer linear expression of the form::

            dst = base + (index << s)

        where `s` is a shift encoded in the instruction.
        Therefore, it may act as a compact *shift-then-add* primitive.

    References:

    * https://forums.developer.nvidia.com/t/how-to-understand-the-lea-assembly-behind-the-cuda-c/257679/2
    """
    SHIFT: typing.Final[str] = r'0x[0-9]+'

    PATTERN: typing.Final[regex.Pattern[str]] = regex.compile(PatternMatcher.build_pattern(
        opcode='LEA',
        modifiers=(),
        operands=(
            Register.REG,
            Register.REG,
            Register.REG,
            PatternBuilder.group(SHIFT, group='shift'),
        ),
        predicate=False,
    ))

    def __init__(self, *,
        dest: str | None = None,
        index: str | None = None,
        base: str | None = None,
        shift: str | None = None,
    ) -> None:
        super().__init__(
            pattern=self.PATTERN
            if not (dest or index or base or shift)
            else self.build_pattern(
                opcode='LEA',
                modifiers=(),
                operands=(
                    dest or Register.REG,
                    index or Register.REG,
                    base or Register.REG,
                    PatternBuilder.group(shift or self.SHIFT, group='shift'),
                ),
                predicate=False,
            ),
        )
