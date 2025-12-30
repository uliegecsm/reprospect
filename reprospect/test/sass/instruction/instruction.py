from __future__ import annotations

import abc
import dataclasses
import os
import sys
import typing

import mypy_extensions
import regex
import semantic_version

from reprospect.test.sass.instruction.operand import Operand
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass.decode import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

SEPARATOR: typing.Final[str] = r'\s*'
"""
Separator between instruction components (between predicate and opcode, or separating
the opcode with its modifiers from the operands).
"""

OPERAND_SEPARATOR: typing.Final[str] = r',?\s*'
"""
Separator between operands.
"""

@dataclasses.dataclass(frozen=True, slots=True)
class ZeroOrOne:
    """
    Mark an instruction component as optional.
    """
    cmpnt: str | int

class Predicate:
    """
    Predicate patterns.
    """
    PREDICATE: typing.Final[str] = r'@!?U?P(?:T|[0-9]+)'
    """
    Predicate for the whole instruction (comes before the opcode).
    """

    @classmethod
    def predicate(cls) -> str:
        """
        :py:attr:`PREDICATE` with `predicate` group.
        """
        return PatternBuilder.group(cls.PREDICATE, group='predicate')

class OpCode:
    """
    Opcode patterns.
    """
    OPCODE: typing.Final[str] = r'[A-Z0-9]+'
    """Opcode."""

    MODIFIER: typing.Final[str] = r'[A-Z0-9_]+'
    """Opcode modifier."""

    @classmethod
    def opcode(cls) -> str:
        """
        :py:attr:`OPCODE` with `opcode` group.
        """
        return PatternBuilder.group(cls.OPCODE, group='opcode')

    @classmethod
    def modifier(cls) -> str:
        """
        :py:attr:`MODIFIER` with `modifiers` group.
        """
        return PatternBuilder.group(cls.MODIFIER, group='modifiers')

    @classmethod
    def mod(cls, opcode: str, modifiers: typing.Iterable[str | int | ZeroOrOne] | None = None, *, captured: bool = True) -> str:
        """
        Append each modifier with a `.` separator.

        Modifiers wrapped in a :py:class:`reprospect.test.sass.instruction.instruction.ZeroOrOne` instance are matched optionally.
        """
        pattern: str = PatternBuilder.group(opcode, group='opcode') if captured else opcode

        if modifiers is not None:
            for mod in modifiers:
                if isinstance(mod, ZeroOrOne):
                    pattern += PatternBuilder.zero_or_one(rf'\.{PatternBuilder.group(str(mod.cmpnt), group="modifiers") if captured else str(mod.cmpnt)}')
                else:
                    pattern += rf'\.{PatternBuilder.group(str(mod), group="modifiers") if captured else str(mod)}'
        else:
            pattern += PatternBuilder.zero_or_more(rf'\.{cls.modifier() if captured else cls.MODIFIER}')

        return pattern

@mypy_extensions.mypyc_attr(native_class=True)
@dataclasses.dataclass(frozen=True, slots=True)
class InstructionMatch:
    """
    An instruction with parsed components.
    """
    opcode: str
    modifiers: tuple[str, ...]
    operands: tuple[str, ...]
    predicate: str | None = None

    additional: dict[str, list[str]] | None = None

    @staticmethod
    def parse(*, bits: regex.Match[str]) -> InstructionMatch:
        """
        The only mandatory capture group is `opcode`.
        """
        captured = bits.capturesdict()

        opcode = captured.pop('opcode')
        assert len(opcode) == 1

        predicate = captured.pop('predicate', None)
        if predicate is not None:
            assert len(predicate) <= 1

        return InstructionMatch(
            opcode=opcode[0],
            modifiers=tuple(captured.pop('modifiers', ())),
            operands=tuple(captured.pop('operands', ())),
            predicate=predicate[0] if (predicate is not None and len(predicate) == 1) else None,
            additional=captured or None,
        )

@mypy_extensions.mypyc_attr(allow_interpreted_subclasses=True)
class InstructionMatcher(abc.ABC):
    """
    Abstract base class for instruction matchers.
    """
    @abc.abstractmethod
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        """
        Check if the instruction matches.
        """

    def __call__(self, inst: Instruction | str) -> InstructionMatch | None:
        """
        Allow the matcher to be called as a function.
        """
        return self.match(inst)

@mypy_extensions.mypyc_attr(allow_interpreted_subclasses=True)
class PatternMatcher(InstructionMatcher):
    """
    Regex-based (or pattern) matching.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('pattern',)

    def __init__(self, pattern: str | regex.Pattern[str]) -> None:
        self.pattern: typing.Final[regex.Pattern[str]] = pattern if isinstance(pattern, regex.Pattern) else regex.compile(pattern)

    @classmethod
    def build_pattern(cls, *,
        opcode: str | None = None,
        modifiers: typing.Iterable[str | int | ZeroOrOne] | None = None,
        operands: typing.Iterable[str | ZeroOrOne] | None = None,
        predicate: str | bool | None = None,
    ) -> str:
        parts: list[str | None] = []

        if isinstance(predicate, str):
            parts.append(PatternBuilder.group(predicate, group='predicate'))
        elif predicate is True:
            parts.append(Predicate.predicate())
        elif predicate is None:
            parts.append(PatternBuilder.zero_or_one(Predicate.predicate()))

        parts.append(OpCode.mod(opcode=opcode or OpCode.OPCODE, modifiers=modifiers))

        parts.append(cls.build_pattern_operands(operands=operands))

        return SEPARATOR.join(filter(None, parts))

    @classmethod
    def build_pattern_operands(cls, operands: typing.Iterable[str | ZeroOrOne] | None = None, *, captured: bool = True) -> str | None:
        """
        Build an operands pattern.

        Join multiple operands with a :py:const:`reprospect.test.sass.instruction.instruction.OPERAND_SEPARATOR` separator.
        Operands wrapped in a :py:class:`reprospect.test.sass.instruction.instruction.ZeroOrOne` instance are matched optionally.
        """
        if operands is not None:
            parts: list[str] = []
            for opnd in operands:
                if not opnd:
                    continue
                if isinstance(opnd, ZeroOrOne):
                    parts.append(PatternBuilder.zero_or_one(PatternBuilder.group(opnd.cmpnt, group="operands") if captured else str(opnd.cmpnt)))
                else:
                    parts.append(PatternBuilder.group(opnd, group="operands") if captured else str(opnd))
            return OPERAND_SEPARATOR.join(parts)
        opnd = Operand.operand() if captured else Operand.OPERAND
        return PatternBuilder.zero_or_one(
            opnd + PatternBuilder.zero_or_more(OPERAND_SEPARATOR + opnd),
        )

    @override
    @typing.final
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        if (matched := self.pattern.match(inst.instruction if isinstance(inst, Instruction) else inst)) is not None:
            return InstructionMatch.parse(bits=matched)
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pattern={self.pattern})'

class ArchitectureAwarePatternMatcher(PatternMatcher):
    """
    Base class for matchers that generate patterns based on architecture.
    """
    def __init__(self, arch: NVIDIAArch) -> None:
        self.arch = arch
        super().__init__(pattern=self._build_pattern())

    @abc.abstractmethod
    def _build_pattern(self) -> str:
        """
        Build the regex pattern based on architecture.
        """

class ArchitectureAndVersionAwarePatternMatcher(ArchitectureAwarePatternMatcher):
    """
    Base class for matchers that generate patterns based on CUDA version and architecture.

    .. note::

        The CUDA version is defaulted to the `CUDA_VERSION` environment variable. However, it must be noted that
        it is expected to be the version of ``ptxas``.

        The version is not always needed, but is useful for some SASS instructions that changed over the course of CUDA ISA evolution.
        For instance, under CUDA 12.6, an atomic add for `int` (at block scope) translates to::

            ATOM.E.ADD.STRONG.CTA PT, RZ, [R2], R5

        whereas for CUDA 12.8.1 or 13.0.0, it translates to::

            ATOM.E.ADD.S32.STRONG.CTA PT, RZ, [R2], R5
    """
    def __init__(self, arch: NVIDIAArch, version: semantic_version.Version | None = None) -> None:
        self.version = version if version is not None else semantic_version.Version(os.environ['CUDA_VERSION'])
        super().__init__(arch=arch)

class OpcodeModsMatcher(PatternMatcher):
    """
    Matcher that will collect all operands of an instruction.

    Useful when the opcode and modifiers are known and the operands may need to be retrieved.

    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
    >>> OpcodeModsMatcher(opcode = 'ISETP', modifiers = ('NE', 'AND')).match(
    ...     'ISETP.NE.AND P2, PT, R4, RZ, PT'
    ... )
    InstructionMatch(opcode='ISETP', modifiers=('NE', 'AND'), operands=('P2', 'PT', 'R4', 'RZ', 'PT'), predicate=None, additional=None)
    """
    def __init__(self, *,
        opcode: str,
        modifiers: typing.Iterable[str | int | ZeroOrOne] | None = None,
        operands: bool = True,
    ) -> None:
        super().__init__(pattern=self.build_pattern(
            opcode=opcode,
            modifiers=modifiers,
            operands=None if operands is True else (),
            predicate=False,
        ))

class OpcodeModsWithOperandsMatcher(PatternMatcher):
    """
    Matcher that matches a given instruction and operands.

    Similar to :py:class:`OpcodeModsMatcher`, but the operands can be better constrained.

    >>> from reprospect.test.sass.instruction import OpcodeModsWithOperandsMatcher, PatternBuilder
    >>> from reprospect.test.sass.instruction.register import Register
    >>> OpcodeModsWithOperandsMatcher(
    ...     opcode = 'ISETP',
    ...     modifiers = ('NE', 'AND'),
    ...     operands = (
    ...         Register.PRED,
    ...         Register.PREDT,
    ...         'R4',
    ...         Register.REGZ,
    ...         Register.PREDT,
    ...     )
    ... ).match('ISETP.NE.AND P2, PT, R4, RZ, PT')
    InstructionMatch(opcode='ISETP', modifiers=('NE', 'AND'), operands=('P2', 'PT', 'R4', 'RZ', 'PT'), predicate=None, additional=None)

    .. note::

        Some operands can be optionally matched.

        >>> from reprospect.test.sass.instruction import OpcodeModsWithOperandsMatcher, PatternBuilder
        >>> from reprospect.test.sass.instruction.instruction import ZeroOrOne
        >>> matcher = OpcodeModsWithOperandsMatcher(opcode = 'WHATEVER', operands = (
        ...     ZeroOrOne('R0'),
        ...     ZeroOrOne('R9'),
        ... ))
        >>> matcher.match('WHATEVER')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=(), predicate=None, additional=None)
        >>> matcher.match('WHATEVER R0')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('R0',), predicate=None, additional=None)
        >>> matcher.match('WHATEVER R0, R9')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('R0', 'R9'), predicate=None, additional=None)
    """
    def __init__(self, *,
        opcode: str,
        modifiers: typing.Iterable[str | int | ZeroOrOne] | None = None,
        operands: typing.Iterable[str | ZeroOrOne] | None = None,
    ) -> None:
        super().__init__(pattern=self.build_pattern(
            opcode=opcode,
            modifiers=modifiers,
            operands=operands,
            predicate=False,
        ))

class AnyMatcher(PatternMatcher):
    """
    Match any instruction.

    .. note::

        The instruction is decomposed into its components.

    .. warning::

        As explained in https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L251-LL258,
        nearly all operands are comma-separated.
        Notable exceptions::

            RET.REL.NODEC R10 0x0

    >>> from reprospect.test.sass.instruction import AnyMatcher
    >>> AnyMatcher().match(inst = 'FADD.FTZ.RN R0, R1, R2')
    InstructionMatch(opcode='FADD', modifiers=('FTZ', 'RN'), operands=('R0', 'R1', 'R2'), predicate=None, additional=None)
    >>> AnyMatcher().match(inst = 'RET.REL.NODEC R4 0x0')
    InstructionMatch(opcode='RET', modifiers=('REL', 'NODEC'), operands=('R4', '0x0'), predicate=None, additional=None)
    """
    PATTERN: typing.Final[regex.Pattern[str]] = regex.compile(PatternMatcher.build_pattern())

    def __init__(self) -> None:
        super().__init__(pattern=self.PATTERN)
