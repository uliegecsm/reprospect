from __future__ import annotations

import abc
import dataclasses
import os
import sys
import types
import typing

import mypy_extensions
import regex
import semantic_version

from reprospect.test.sass.instruction.address import AddressMatcher
from reprospect.test.sass.instruction.constant import Constant, ConstantMatcher
from reprospect.test.sass.instruction.immediate import Immediate
from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.test.sass.instruction.operand import Operand
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

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
        Append each modifier with a `.` separator. Modifiers wrapped in a :py:class:`ZeroOrOne` instance are matched optionally.
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

    @override
    @typing.final
    def match(self, inst: Instruction | str) -> InstructionMatch | None:
        if (matched := self.pattern.match(inst.instruction if isinstance(inst, Instruction) else inst)) is not None:
            return InstructionMatch.parse(bits=matched)
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pattern={self.pattern})'

def floating_point_add_pattern(*, ftype: typing.Literal['F', 'D']) -> regex.Pattern[str]:
    """
    Helper for:

    * :py:class:reprospect.test.sass.instruction.Fp32AddMatcher`.
    * :py:class:reprospect.test.sass.instruction.Fp64AddMatcher`.
    """
    return regex.compile(
        OpCode.mod(f'{ftype}ADD', modifiers=(ZeroOrOne('FTZ'),)) + r'\s+' +
        Register.dst() + r'\s*,\s*' +
        Operand.mod(Register.REGZ, math=None) + r'\s*,\s*' +
        PatternBuilder.any(
            Operand.mod(Register.REGZ, math=None), Register.ureg(),
            Constant.address(),
            Immediate.floating(),
        ),
    )

class Fp32AddMatcher(PatternMatcher):
    """
    Matcher for 32-bit floating-point add (``FADD``) instructions.
    """
    PATTERN: typing.Final[regex.Pattern[str]] = floating_point_add_pattern(ftype='F')

    def __init__(self) -> None:
        super().__init__(pattern=self.PATTERN)

class Fp64AddMatcher(PatternMatcher):
    """
    Matcher for 64-bit floating-point add (``DADD``) instructions.
    """
    PATTERN: typing.Final[regex.Pattern[str]] = floating_point_add_pattern(ftype='D')

    def __init__(self) -> None:
        super().__init__(pattern=self.PATTERN)

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

ExtendBitsMethod: typing.TypeAlias = typing.Literal['U', 'S']
"""
How bits must be extended, see https://www.cs.fsu.edu/~hawkes/cda3101lects/chap4/extension.htm.

`U` and `S` stand for *zero extension* and *sign extension*, respectively.
"""

def check_memory_instruction_word_size(*, size: int) -> None:
    """
    From https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses:

        Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes.
    """
    ALLOWABLE_SIZES: typing.Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32) # pylint: disable=invalid-name
    if size not in ALLOWABLE_SIZES:
        raise RuntimeError(f'{size} is not an allowable memory instruction word size ({ALLOWABLE_SIZES} (in bytes)).')

class LoadMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for load instructions, such as::

        LDG.E R2, desc[UR6][R2.64]
        LD.E.64 R2, R4.64

    Starting from `BLACKWELL`, 256-bit load instructions are available, such as::

        LDG.E.ENL2.256.CONSTANT R12, R8, desc[UR4][R2.64]

    References:

    * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldg#global-memory-5-x
    * https://github.com/AdaptiveCpp/AdaptiveCpp/issues/848
    * https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsglobal.htm
    * https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/#updated_vector_types
    """
    __slots__ = ('cache', 'extend', 'memory', 'size')

    TEMPLATE:     typing.Final[str] = f'{{opcode}} {Register.reg()}, {{address}}'
    TEMPLATE_256: typing.Final[str] = f'{{opcode}} {Register.reg()}, {Register.reg()}, {{address}}'

    def __init__(self,
        arch: NVIDIAArch,
        *,
        size: int | None = None,
        readonly: bool | None = None,
        memory: MemorySpace | str = MemorySpace.GLOBAL,
        extend: ExtendBitsMethod | None = None,
    ):
        """
        :param size: Optional bit size (*e.g.*, 32, 64, 128).
        :param readonly: Whether to append ``.CONSTANT`` modifier. If `None`, the modifier is matched optionally.
        """
        if size is not None:
            check_memory_instruction_word_size(size=size // 8)

        self.size: int | None = size
        self.cache: str | ZeroOrOne | None = None if readonly is False else ('CONSTANT' if readonly is True else ZeroOrOne('CONSTANT'))
        self.memory: typing.Final[MemorySpace] = MemorySpace(memory)
        self.extend: typing.Final[ExtendBitsMethod | None] = extend

        super().__init__(arch=arch)

    def _get_size(self) -> int | str | None:
        if self.size is not None and self.size < 32:
            return f'{self.extend}{self.size}'
        if self.size is not None and self.size > 32:
            return self.size
        return None

    def _get_modifiers(self) -> typing.Iterable[str | int | ZeroOrOne]:
        return filter(None, (
            'E',
            *(('ENL2',) if self.size is not None and self.size == 256 else ()),
            self._get_size(),
            self.cache,
            *(('SYS',) if self.arch.compute_capability.as_int in {70, 75} else ()),
        ))

    @override
    def _build_pattern(self) -> str:
        return (self.TEMPLATE_256 if self.size is not None and self.size == 256 else self.TEMPLATE).format(
            opcode=OpCode.mod(
                opcode=f'LD{self.memory.value}',
                modifiers=self._get_modifiers(),
            ),
            address=PatternBuilder.groups(AddressMatcher.build_pattern(arch=self.arch, memory=self.memory), groups=('operands', 'address')),
        )

class LoadGlobalMatcher(LoadMatcher):
    """
    Specialization of :py:class:`LoadMatcher` for global memory (``LDG``).
    """
    def __init__(self, arch: NVIDIAArch, *, size: int | None = None, readonly: bool | None = None, extend: ExtendBitsMethod | None = None) -> None:
        super().__init__(arch=arch, size=size, readonly=readonly, memory=MemorySpace.GLOBAL, extend=extend)

class LoadConstantMatcher(PatternMatcher):
    """
    Matcher for constant load (``LDC``) instructions, like:

    * ``LDC.64 R2, c[0x0][0x388]``
    * ``LDC R4, c[0x3][R0]``
    * ``LDCU UR4, c[0x3][UR0]``
    """
    CONSTANT: typing.Final[str] = ConstantMatcher.build_pattern(captured=True, capture_bank=True, capture_offset=True)

    TEMPLATE: typing.Final[str] = f'{{opcode}} {{dest}}, {CONSTANT}'

    def __init__(self, *, uniform: bool | None = None, size: int | None = None) -> None:
        """
        :param size: Optional bit size (*e.g.*, 32, 64, 128).
        :param uniform: Optionally require uniformness.
        """
        if size is not None:
            check_memory_instruction_word_size(size=size // 8)

        if uniform is None:
            opcode = PatternBuilder.any('LDC', 'LDCU')
            dest   = PatternBuilder.any(Register.REG, Register.UREG)
        elif uniform is True:
            opcode = 'LDCU'
            dest   = Register.UREG
        else:
            opcode = 'LDC'
            dest   = Register.REG

        super().__init__(pattern=self.TEMPLATE.format(
            opcode=OpCode.mod(
                opcode=opcode,
                modifiers=(size,) if size else (),
            ),
            dest=PatternBuilder.group(dest, group='operands'),
        ))

class StoreMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for global store instructions, such as::

        STG.E desc[UR6][R6.64], R15
        ST.E.64 R4.64, R2

    Starting from `BLACKWELL`, 256-bit store instructions are available, such as::

        STG.E.ENL2.256 desc[UR4][R4.64], R8, R12
    """
    __slots__ = ('extend', 'memory', 'size')

    TEMPLATE:     typing.Final[str] = f'{{opcode}} {{address}}, {Register.reg()}'
    TEMPLATE_256: typing.Final[str] = f'{{opcode}} {{address}}, {Register.reg()}, {Register.reg()}'

    def __init__(self,
        arch: NVIDIAArch,
        size: int | None = None,
        memory: MemorySpace | str = MemorySpace.GLOBAL,
        extend: ExtendBitsMethod | None = None,
    ):
        """
        :param size: Optional bit size (*e.g.*, 32, 64, 128).
        """
        if size is not None:
            check_memory_instruction_word_size(size=size // 8)

        self.size = size
        self.memory: typing.Final[MemorySpace] = MemorySpace(memory)
        self.extend: typing.Final[ExtendBitsMethod | None] = extend
        super().__init__(arch=arch)

    def _get_size(self) -> int | str | None:
        if self.size is not None and self.size < 32:
            return f'{self.extend}{self.size}'
        if self.size is not None and self.size > 32:
            return self.size
        return None

    def _get_modifiers(self) -> typing.Iterable[str | int]:
        return filter(None, (
            'E',
            *(('ENL2',) if self.size is not None and self.size == 256 else ()),
            self._get_size(),
            *(('SYS',) if self.arch.compute_capability.as_int in {70, 75} else ()),
        ))

    @override
    def _build_pattern(self) -> str:
        return (self.TEMPLATE_256 if self.size is not None and self.size == 256 else self.TEMPLATE).format(
            opcode=OpCode.mod(
                opcode=f'ST{self.memory}',
                modifiers=self._get_modifiers(),
            ),
            address=PatternBuilder.groups(AddressMatcher.build_pattern(arch=self.arch, memory=self.memory), groups=('operands', 'address')),
        )

class StoreGlobalMatcher(StoreMatcher):
    """
    Specialization of :py:class:`StoreMatcher` for global memory (``STG``).
    """
    def __init__(self, arch: NVIDIAArch, size: int | None = None, extend: ExtendBitsMethod | None = None) -> None:
        super().__init__(arch=arch, size=size, memory=MemorySpace.GLOBAL, extend=extend)

ThreadScope = typing.Literal['BLOCK', 'DEVICE', 'THREADS']
"""
References:

* https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes
"""

def convert_thread_scope(*, scope: ThreadScope, arch: NVIDIAArch) -> str:
    """
    Convert the scope to SASS modifier.

    For instance, ``__NV_THREAD_SCOPE_DEVICE`` maps to ``GPU``.

    References:

    * https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions

    .. warning::

        ``__NV_THREAD_SCOPE_THREAD`` thread scope is currently implemented using wider ``__NV_THREAD_SCOPE_BLOCK`` thread scope.
    """
    match scope:
        case 'THREADS' | 'BLOCK':
            if arch.compute_capability < 80:
                return 'CTA'
            return 'SM'
        case 'DEVICE':
            return 'GPU'
        case _:
            raise ValueError(f'unsupported {scope}')

class ReductionMatcher(ArchitectureAwarePatternMatcher):
    """
    Matcher for reduction operations on generic memory (``RED``).

    ``RED`` instructions are typically used when the atomic operation return value is not used.
    Otherwise, it would typically map to ``ATOM``.

    The ``RED`` opcode may take several modifiers:

    * operation (*e.g.* `ADD`)
    * scope (*e.g.* `DEVICE`)
    * consistency (*e.g.* `STRONG`)

    References:

    * https://forums.developer.nvidia.com/t/difference-between-red-and-atomg-sass-instruction/203469
    """
    TEMPLATE: typing.Final[str] = f'{{opcode}} {{address}}, {Register.reg()}'

    def __init__(self,
        arch: NVIDIAArch,
        operation: str = 'ADD',
        scope: ThreadScope | None = None,
        consistency: str = 'STRONG',
        dtype: tuple[str, int] | None = None,
    ):
        """
        :param dtype: For instance, `('F', 64)` for a 64-bit floating-point or `(S, 32)` for a signed 32-bit integer.
        """
        if dtype is not None:
            check_memory_instruction_word_size(size=int(dtype[1] / 8))

        self.params = types.SimpleNamespace(
            operation=operation,
            scope=convert_thread_scope(scope=scope, arch=arch) if scope else None,
            consistency=consistency,
            dtype=dtype,
        )
        super().__init__(arch=arch)

    @override
    def _build_pattern(self) -> str:
        dtype: typing.Sequence[int | str | ZeroOrOne]
        if self.params.dtype is not None:
            match self.params.dtype[0]:
                case 'F':
                    dtype = (
                        f'{self.params.dtype[0]}{self.params.dtype[1]}',
                        ZeroOrOne('FTZ'),
                        ZeroOrOne('RN'),
                    )
                case 'S':
                    if self.params.operation == 'ADD':
                        dtype = ()
                    else:
                        dtype = (f'{self.params.dtype[0]}{self.params.dtype[1]}',)
                case 'U':
                    dtype = (self.params.dtype[1],) if self.params.dtype[1] > 32 else ()
                case _:
                    raise ValueError(self.params.dtype)
        else:
            dtype = ()

        match self.arch.compute_capability.as_int:
            case 70 | 75 | 80 | 86 | 89:
                opcode = 'RED'
            case 90 | 100 | 103 | 120:
                opcode = 'REDG'
            case _:
                raise ValueError(f'unsupported {self.arch}')

        return self.TEMPLATE.format(
            opcode=OpCode.mod(
                opcode=opcode,
                modifiers=filter(None, ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)),
            ),
            address=PatternBuilder.groups(AddressMatcher.build_pattern(arch=self.arch), groups=('operands', 'address')),
        )

class AtomicMatcher(ArchitectureAndVersionAwarePatternMatcher):
    """
    Matcher for atomic operations on:

    * generic memory (``ATOM``)
    * shared memory (``ATOMS``)
    * global memory (``ATOMG``)

    Unlike ``RED``, these operations capture the return value.

    The ``ATOM`` opcode may take several modifiers:

    * operation (*e.g.* `CAS`)
    * scope (*e.g.* `DEVICE`)
    * consistency (*e.g.* `STRONG`)

    References:

    * https://docs.nvidia.com/cuda/archive/12.6.3/cuda-binary-utilities/index.html#hopper-instruction-set
    """
    TEMPLATE_CAS: typing.Final[str] = rf'{{opcode}} {Register.predt()}, {Register.regz()}, {{address}}, {Register.reg()}, {Register.reg()}'
    TEMPLATE: typing.Final[str]     = rf'{{opcode}} {Register.predt()}, {Register.regz()}, {{address}}, {Register.regz()}'

    def __init__(self,
        arch: NVIDIAArch,
        operation: str = 'ADD',
        scope: ThreadScope | None = None,
        consistency: str = 'STRONG',
        dtype: tuple[str | None, int] | None = None,
        memory: MemorySpace | str = MemorySpace.GLOBAL,
        version: semantic_version.Version | None = None,
    ):
        """
        :param dtype: For instance, `('F', 64)` for a 64-bit floating-point or `(S, 32)` for a signed 32-bit integer.
        """
        if dtype is not None:
            check_memory_instruction_word_size(size=int(dtype[1] / 8))

        self.params = types.SimpleNamespace(
            operation=operation,
            scope=convert_thread_scope(scope=scope, arch=arch) if scope else None,
            consistency=consistency,
            memory=MemorySpace(memory),
            dtype=dtype,
        )
        super().__init__(arch=arch, version=version)

    @override
    def _build_pattern(self) -> str: # pylint: disable=too-many-branches
        """
        ``CAS`` has a different operand structure. For instance::

            ATOMG.E.CAS.STRONG.GPU PT, R7, [R2], R6, R7

        The generic pattern for the operands is thus::

            {pred}, {dest}, [{addr}], {compare}, {newval}
        """
        dtype: typing.Sequence[int | str | ZeroOrOne]
        match self.params.operation:
            case 'CAS':
                dtype = (self.params.dtype[1],) if self.params.dtype is not None and self.params.dtype[1] > 32 else ()
            case _:
                if self.params.dtype is not None and self.params.operation == 'EXCH':
                    dtype = (self.params.dtype[1],) if self.params.dtype[1] > 32 else ()
                elif self.params.dtype is not None:
                    match self.params.dtype[0]:
                        case 'F':
                            dtype = (
                                f'{self.params.dtype[0]}{self.params.dtype[1]}',
                                ZeroOrOne('FTZ'),
                                ZeroOrOne('RN'),
                            )
                        case 'S':
                            if self.params.operation == 'ADD' and self.version in semantic_version.SimpleSpec('<12.8'):
                                dtype = ()
                            else:
                                dtype = (f'{self.params.dtype[0]}{self.params.dtype[1]}',)
                        case 'U':
                            dtype = (self.params.dtype[1],) if self.params.dtype[1] > 32 else ()
                        case _:
                            raise ValueError(self.params.dtype)
                else:
                    dtype = ()

        address: str = AddressMatcher.build_pattern(arch=self.arch, memory=self.params.memory)

        match self.arch.compute_capability.as_int:
            case 80 | 86 | 89 | 90 | 100 | 103 | 120:
                address = PatternBuilder.any(AddressMatcher.build_reg_address(arch=self.arch), address)
            case _:
                pass

        return (self.TEMPLATE_CAS if self.params.operation == 'CAS' else self.TEMPLATE).format(
            opcode=OpCode.mod(
                opcode=f'ATOM{self.params.memory}',
                modifiers=filter(None, ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)),
            ),
            address=PatternBuilder.groups(address, groups=('operands', 'address')),
        )

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
        pattern = OpCode.mod(opcode, modifiers)

        if operands:
            pattern_operands = PatternBuilder.zero_or_one(
                Operand.operand() + PatternBuilder.zero_or_more(r'\s*,\s*' + Operand.operand()),
            )
            pattern += r'\s+' + pattern_operands

        super().__init__(pattern=pattern)

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
        >>> matcher = OpcodeModsWithOperandsMatcher(opcode = 'WHATEVER', operands = (
        ...     PatternBuilder.zero_or_one('R0'),
        ...     PatternBuilder.zero_or_one('R9'),
        ... ))
        >>> matcher.match('WHATEVER')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('',), predicate=None, additional=None)
        >>> matcher.match('WHATEVER R0')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('R0',), predicate=None, additional=None)
        >>> matcher.match('WHATEVER R0, R9')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('R0', 'R9'), predicate=None, additional=None)
    """
    SEPARATOR: typing.Final[str] = r',\s+'

    @classmethod
    def operand(cls, *, op: str) -> str:
        pattern = PatternBuilder.group(op, group='operands')
        if op.startswith('(') and op.endswith(')?'):
            return PatternBuilder.zero_or_more(cls.SEPARATOR + pattern)
        return cls.SEPARATOR + pattern

    def __init__(self, *,
        opcode: str,
        operands: typing.Iterable[str],
        modifiers: typing.Iterable[str | int | ZeroOrOne] | None = None,
    ) -> None:
        ops_iter = iter(operands)
        ops: str = PatternBuilder.group(next(ops_iter), group='operands')
        ops += ''.join(self.operand(op=op) for op in ops_iter)
        super().__init__(pattern=rf'{OpCode.mod(opcode, modifiers)}\s*{ops}')

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
    TEMPLATE: typing.Final[str] = r'{predicate}{opcode}{modifiers}\s*{operands}{operand}'

    PATTERN: typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(
        predicate=PatternBuilder.zero_or_one(Predicate.predicate() + r'\s*'),
        opcode=   PatternBuilder.group(r'[A-Z0-9]+', group='opcode'),  # noqa: E251
        modifiers=PatternBuilder.zero_or_more(r'\.' + PatternBuilder.group(s=r'[A-Z0-9_]+', group='modifiers')),
        operands= PatternBuilder.zero_or_more(PatternBuilder.group(s=r'[^,\s]+', group='operands') + PatternBuilder.any(r'\s*,\s*', r'\s+')),  # noqa: E251
        operand=  PatternBuilder.zero_or_one(PatternBuilder.group(s=r'[^,\s]+', group='operands')),  # noqa: E251
    ))

    def __init__(self):
        super().__init__(pattern=self.PATTERN)

class BranchMatcher(PatternMatcher):
    """
    Matcher for a ``BRA`` branch instruction.

    Typically::

        @!UP5 BRA 0x456
    """
    BRA: typing.Final[str] = rf"{OpCode.mod('BRA', modifiers=())}\s*{PatternBuilder.hex()}"

    PREDICATE: typing.Final[str] = PatternBuilder.zero_or_one(Predicate.predicate())

    TEMPLATE: typing.Final[str] = rf'{{predicate}}\s*{BRA}'

    PATTERN: typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(predicate=PREDICATE))

    def __init__(self, predicate: str | None = None):
        super().__init__(pattern=self.PATTERN if predicate is None else self.TEMPLATE.format(predicate=PatternBuilder.group(predicate, group='predicate')))
