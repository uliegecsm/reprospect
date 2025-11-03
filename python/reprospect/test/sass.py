"""
At first glance, examining generated SASS code may appear to be an esoteric task suited only to expert lab researchers — let alone testing it.

Yet, studying SASS — and assembly code in general — offers valuable insights.
Indeed, modern HPC code bases rely on complex software stacks and compiler toolchains.
While code correctness is often checked through regression testing,
reaching and sustaining optimal performance as software and hardware evolve requires additional effort.
This is usually achieved through verification of compile flags and *ad hoc* profiling and benchmarking.
However, beyond runtime analysis, the SASS code already contains information about the code paths taken and could itself be incorporated into testing.
Still, the barrier to entry for meaningful SASS analysis is high: results can vary dramatically with compiler versions, optimization flags, and target architectures.

`ReProspect` provides a hierarchy of SASS instruction matchers that capture the components of an instruction (*opcodes*, *modifiers* and *operands*).
Under the hood, they generate complex regular expression (*regex*) patterns.
To accommodate for the evolving CUDA Instruction Set,
some of these matchers take the target architecture as a parameter and adjust the regex patterns accordingly.
In this way, `ReProspect` helps developers write assertions about expected instructions,
while reducing the need to track low-level details of the evolving CUDA instruction set.

.. doctest::

    >>> from reprospect.tools.architecture import NVIDIAArch
    >>> from reprospect.test.sass          import LoadGlobalMatcher
    >>> LoadGlobalMatcher(arch = NVIDIAArch.from_str('VOLTA70')).matches(inst = 'LDG.E.SYS R15, [R8+0x10]').capturesdict()
    {'opcode': ['LDG'], 'modifiers': ['E', 'SYS'], 'operands': ['R15', 'R8+0x10'], 'address': ['R8+0x10']}
    >>> LoadGlobalMatcher(arch = NVIDIAArch.from_str('BLACKWELL120'), size = 128, readonly = True).matches(inst = 'LDG.E.128.CONSTANT R2, desc[UR15][R6.64+0x12]').capturesdict()
    {'opcode': ['LDG'], 'modifiers': ['E', '128', 'CONSTANT'], 'operands': ['R2', 'desc[UR15][R6.64+0x12]'], 'address': ['desc[UR15][R6.64+0x12]']}

References:

* https://github.com/Usibre/asmregex
* :cite:`yan-optimizing-winograd-2020`
* https://github.com/0xD0GF00D/DocumentSASS
* https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L9C75-L9C83
"""

import abc
import dataclasses
import functools
import os
import sys
import types
import typing

import regex
import semantic_version
import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass         import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class PatternBuilder:
    """
    Helper class to build patterns for instruction components.
    """
    OFFSET = r'\+0x[0-9A-Fa-f]+'
    PRED = r'P[0-9]+'
    REG = r'R[0-9]+'
    REG64 = r'R[0-9]+\.64'
    UREG = r'UR[0-9]+'

    #: Match a register or ``RZ``.
    REGZ = rf'RZ|{REG}'

    #: Match a predicate register or ``PT``.
    PREDT = rf'PT|{PRED}'

    OPERAND = r'[\w@!\.\[\]\+\-\s]+'

    CONSTANT = rf'c\[0x[0-9]+\]\[(0x[0-9c]+|{REG})\]'
    """
    Match constant memory location.
    The bank looks like ``0x3`` while the address is either compile-time (*e.g.*
    ``0x899``) or depends on a register.
    """

    @staticmethod
    @typeguard.typechecked
    def optional(s : str) -> str:
        """
        Build optional non-capturing pattern.
        """
        return rf'(?:{s})?'

    @staticmethod
    @typeguard.typechecked
    def either(a : str, b : str) -> str:
        """
        Build a pattern matching either `a` or `b`.
        """
        return rf'({a}|{b})'

    @staticmethod
    @typeguard.typechecked
    def group(s : typing.Any, *, group : typing.Optional[str] = None, groups : typing.Optional[typing.Iterable[str]] = None) -> str:
        """
        Wrap a pattern in named capture group(s).
        """
        if group and groups:
            raise ValueError("Cannot specify both 'group' and 'groups'.")
        if group:
            return rf'(?P<{group}>{s})'
        if groups:
            for g in groups:
                s = PatternBuilder.group(s, group = g)
        return s

    @classmethod
    @typeguard.typechecked
    def reg(cls) -> str:
        """
        :py:attr:`REG` with `operands` group.
        """
        return cls.group(cls.REG, group = 'operands')

    @classmethod
    @typeguard.typechecked
    def regz(cls) -> str:
        """
        :py:attr:`REGZ` with `operands` group.
        """
        return cls.group(cls.REGZ, group = 'operands')

    @classmethod
    @typeguard.typechecked
    def predt(cls) -> str:
        """
        :py:attr:`PREDT` with `operands` group.
        """
        return cls.group(cls.PREDT, group = 'operands')

    @staticmethod
    @typeguard.typechecked
    def opcode_mods(opcode : str, modifiers : typing.Optional[typing.Iterable[int | str]] = None) -> str:
        """
        Append each modifier with a `.`, within a proper named capture group.

        Note that the modifiers starting with a `?` are matched optionally.
        """
        opcode = PatternBuilder.group(opcode, group = 'opcode')
        if modifiers:
            for modifier in filter(None, modifiers):
                if isinstance(modifier, str) and modifier.startswith('?'):
                    opcode += PatternBuilder.optional(r'\.' + PatternBuilder.group(modifier[1::], group = 'modifiers'))
                else:
                    opcode += r'\.' + PatternBuilder.group(modifier, group = 'modifiers')
        return opcode

    @classmethod
    @functools.cache
    def operands(cls) -> str:
        """
        Many operands, with the `operands` named capture group.
        """
        op = PatternBuilder.group(cls.OPERAND, group = 'operands')
        return rf'{op}(?:\s*,\s*{op})*'

    @classmethod
    @typeguard.typechecked
    def reg_addr(cls, offset : typing.Optional[bool] = None) -> str:
        """
        Address operand. A few examples:

        * ``[R0]``
        * ``[R2+0x10]``

        :param offset: If not given, the offset matching is optional. It `True`, there must be an offset. If `False`, there must not be an offset.
        """
        pattern = cls.REG
        if offset is None:
            pattern += cls.optional(cls.OFFSET)
        elif offset is True:
            pattern += cls.OFFSET
        return rf"\[{cls.group(pattern, groups = ('operands', 'address'))}\]"

    @classmethod
    @typeguard.typechecked
    def reg64_addr(cls, offset : typing.Optional[bool] = None) -> str:
        """
        Address operand with 64-bit modifier. A few examples:

        * ``[R0.64]``
        * ``[R0.64+0x10]``

        :param offset: If not given, the offset matching is optional. It `True`, there must be an offset. If `False`, there must not be an offset.
        """
        pattern = cls.REG64
        if offset is None:
            pattern += cls.optional(cls.OFFSET)
        elif offset is True:
            pattern += cls.OFFSET
        return rf"\[({cls.group(pattern, groups = ('operands', 'address'))})\]"

    @classmethod
    @typeguard.typechecked
    def desc_reg64_addr(cls, offset : typing.Optional[bool] = None) -> str:
        """
        Address operand with cache policy descriptor, like ``desc[UR0][R0.64+0x10]``.

        :param offset: If not given, the offset matching is optional. It `True`, there must be an offset. If `False`, there must not be an offset.
        """
        pattern = rf'desc\[({cls.UREG})\]'
        if offset is None:
            pattern += rf'\[({cls.REG64}{cls.optional(cls.OFFSET)})\]'
        elif offset is True:
            pattern += rf'\[({cls.REG64}{cls.OFFSET})\]'
        else:
            pattern += rf'\[({cls.REG64})\]'
        return cls.group(pattern, groups = ('operands', 'address'))

@typeguard.typechecked
def check_memory_instruction_word_size(*, size : int) -> None:
    """
    From https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses:

        Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes.
    """
    ALLOWABLE_SIZES : tuple[int, ...] = (1, 2, 4, 8, 16) # pylint: disable=invalid-name
    if size not in ALLOWABLE_SIZES:
        raise RuntimeError(f'{size} is not an allowable memory instruction word size ({ALLOWABLE_SIZES} (in bytes)).')

class Matcher(abc.ABC):
    """
    Abstract base class for instruction matchers.
    """
    @abc.abstractmethod
    @typeguard.typechecked
    def matches(self, inst : Instruction | str) -> typing.Any:
        """
        Check if the instruction matches.
        """

    @typeguard.typechecked
    def __call__(self, inst : Instruction | str) -> typing.Any:
        """
        Allow the matcher to be called as a function.
        """
        return self.matches(inst)

@dataclasses.dataclass(slots = True, kw_only = True)
class PatternMatcher(Matcher):
    """
    Regex-based (or pattern) matching.
    """
    pattern : str | regex.Pattern

    @override
    @typeguard.typechecked
    def matches(self, inst : Instruction | str) -> typing.Optional[regex.Match]:
        if isinstance(inst, str):
            return regex.match(self.pattern, inst)
        return regex.match(self.pattern, inst.instruction)

class FloatAddMatcher(PatternMatcher):
    """
    Matcher for floating-point add (``FADD``) instructions.
    """
    PATTERN = regex.compile(
        PatternBuilder.opcode_mods('FADD') + r'\s+' +
        PatternBuilder.group(PatternBuilder.REG, groups = ('dst', 'operands')) + r'\s*,\s*' +
        PatternBuilder.reg() + r'\s*,\s*' +
        PatternBuilder.reg()
    )

    def __init__(self) -> None:
        super().__init__(pattern = self.PATTERN)

class ArchitectureAwarePatternMatcher(PatternMatcher):
    """
    Base class for matchers that generate patterns based on architecture.
    """
    @typeguard.typechecked
    def __init__(self, arch : NVIDIAArch) -> None:
        self.arch = arch
        super().__init__(pattern = self._build_pattern())

    @abc.abstractmethod
    def _build_pattern(self) -> str:
        """
        Build the regex pattern based on architecture.
        """

class VersionAwarePatternMixin:
    """
    Base class for matchers that generate patterns based on CUDA version.

    .. note::

        The CUDA version is defaulted to the `CUDA_VERSION` environment variable. However, it must be noted that
        it is expected to be the version of ``ptxas``.

        The version is not always needed, but is useful for some SASS instructions that changed over the course of CUDA ISA evolution.
        For instance, under CUDA 12.6, an atomic add for `int` (at block scope) translates to::

            ATOM.E.ADD.STRONG.CTA PT, RZ, [R2], R5

        whereas for CUDA 12.8.1 or 13.0.0, it translates to::

            ATOM.E.ADD.S32.STRONG.CTA PT, RZ, [R2], R5
    """
    @typeguard.typechecked
    def __init__(self, version : typing.Optional[semantic_version.Version] = None) -> None:
        self.version = version if version is not None else semantic_version.Version(os.environ['CUDA_VERSION'])

class LoadGlobalMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for global load (``LDG``) instructions,
    like ``LDG.E R2, desc[UR6][R2.64]``.

    References:

    * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldg#global-memory-5-x
    * https://github.com/AdaptiveCpp/AdaptiveCpp/issues/848
    * https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsglobal.htm
    """
    @typeguard.typechecked
    def __init__(self,
        arch : NVIDIAArch,
        size : typing.Optional[int] = None,
        readonly : typing.Optional[bool] = None,
    ):
        """
        :param size: Optional bit size (e.g., 32, 64, 128).
        :param readonly: Whether to append ``.CONSTANT`` modifier. If `None`, the modifier is matched optionally.
        """
        if size is not None:
            check_memory_instruction_word_size(size = int(size / 8))

        if readonly is True:
            cache = 'CONSTANT'
        elif readonly is None:
            cache = '?CONSTANT'
        else:
            cache = None
        self.params = types.SimpleNamespace(size = size, cache = cache)
        super().__init__(arch = arch)

    @override
    @typeguard.typechecked
    def _build_pattern(self) -> str:
        match self.arch.compute_capability.as_int:
            case 70 | 75:
                return PatternBuilder.opcode_mods('LDG', ('E', self.params.size, self.params.cache, 'SYS')) + f' {PatternBuilder.reg()}, {PatternBuilder.reg_addr()}'
            case 80 | 86 | 89:
                return PatternBuilder.opcode_mods('LDG', ('E', self.params.size, self.params.cache)) + f' {PatternBuilder.reg()}, {PatternBuilder.reg64_addr()}'
            case 90 | 100 | 120:
                return PatternBuilder.opcode_mods('LDG', ('E', self.params.size, self.params.cache)) + f' {PatternBuilder.reg()}, {PatternBuilder.desc_reg64_addr()}'
            case _:
                raise ValueError(f'unsupported {self.arch}')

class StoreGlobalMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for global store (``STG``) instructions,
    like ``STG.E desc[UR6][R6.64], R15``.
    """
    @typeguard.typechecked
    def __init__(self,
        arch : NVIDIAArch,
        size : typing.Optional[int] = None,
    ):
        """
        :param size: Optional bit size (e.g., 32, 64, 128).
        """
        if size is not None:
            check_memory_instruction_word_size(size = int(size / 8))

        self.params = types.SimpleNamespace(size = size)
        super().__init__(arch = arch)

    @override
    @typeguard.typechecked
    def _build_pattern(self) -> str:
        match self.arch.compute_capability.as_int:
            case 70 | 75:
                return PatternBuilder.opcode_mods('STG', ('E', self.params.size, 'SYS')) + f' {PatternBuilder.reg_addr()}, {PatternBuilder.reg()}'
            case 80 | 86 | 89:
                return PatternBuilder.opcode_mods('STG', ('E', self.params.size)) + f' {PatternBuilder.reg64_addr()}, {PatternBuilder.reg()}'
            case 90 | 100 | 120:
                return PatternBuilder.opcode_mods('STG', ('E', self.params.size)) + f' {PatternBuilder.desc_reg64_addr()}, {PatternBuilder.reg()}'
            case _:
                raise ValueError(f'unsupported {self.arch}')

ThreadScope = typing.Literal['BLOCK', 'DEVICE', 'THREADS']
"""
References:

* https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes
"""

@typeguard.typechecked
def convert_thread_scope(*, scope : ThreadScope, arch : NVIDIAArch) -> str:
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
    @typeguard.typechecked
    def __init__(self,
        arch : NVIDIAArch,
        operation : str = 'ADD',
        scope : typing.Optional[ThreadScope] = None,
        consistency : str = 'STRONG',
        dtype : typing.Optional[tuple[str, int]] = None,
    ):
        """
        :param dtype: For instance, `('F', 64')` for a floating-point type 64-bits in size or `(S, 32)` for a signed integer type 32-bits in size.
        """
        if dtype is not None:
            check_memory_instruction_word_size(size = int(dtype[1] / 8))

        self.params = types.SimpleNamespace(
            operation = operation,
            scope = convert_thread_scope(scope = scope, arch = arch) if scope else None,
            consistency = consistency,
            dtype = dtype,
        )
        super().__init__(arch = arch)

    @override
    def _build_pattern(self) -> str:
        dtype : typing.Sequence[int | str] = ()
        if self.params.dtype is not None:
            match self.params.dtype[0]:
                case 'F':
                    dtype = (
                        f'{self.params.dtype[0]}{self.params.dtype[1]}',
                        '?FTZ',
                        '?RN',
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

        match self.arch.compute_capability.as_int:
            case 70 | 75:
                return PatternBuilder.opcode_mods('RED', ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)) + f' {PatternBuilder.reg_addr()}, {PatternBuilder.reg()}'
            case 80 | 86 | 89:
                return PatternBuilder.opcode_mods('RED', ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)) + f' {PatternBuilder.reg64_addr()}, {PatternBuilder.reg()}'
            case 90 | 100 | 120:
                return PatternBuilder.opcode_mods('REDG', ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)) + f' {PatternBuilder.desc_reg64_addr()}, {PatternBuilder.reg()}'
            case _:
                raise ValueError(f'unsupported {self.arch}')

class AtomicMatcher(VersionAwarePatternMixin, ArchitectureAwarePatternMatcher):
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
    @typeguard.typechecked
    def __init__(self,
        arch : NVIDIAArch,
        operation : str = 'ADD',
        scope : typing.Optional[ThreadScope] = None,
        consistency : str = 'STRONG',
        dtype : typing.Optional[tuple[typing.Optional[str], int]] = None,
        memory : str = 'G',
        version : typing.Optional[semantic_version.Version] = None,
    ):
        """
        :param dtype: For instance, `('F', 64')` for a floating-point type 64-bits in size or `(S, 32)` for a signed integer type 32-bits in size.
        """
        if dtype is not None:
            check_memory_instruction_word_size(size = int(dtype[1] / 8))

        self.params = types.SimpleNamespace(
            operation = operation,
            scope = convert_thread_scope(scope = scope, arch = arch) if scope else None,
            consistency = consistency,
            memory = memory,
            dtype = dtype,
        )
        VersionAwarePatternMixin.__init__(self, version = version)
        ArchitectureAwarePatternMatcher.__init__(self, arch = arch)

    @override
    def _build_pattern(self) -> str: # pylint: disable=too-many-branches
        """
        ``CAS`` has a different operand structure. For instance::

            ATOMG.E.CAS.STRONG.GPU PT, R7, [R2], R6, R7

        The generic pattern for the operands is thus::

            {pred}, {dest}, [{addr}], {compare}, {newval}
        """
        dtype : typing.Sequence[int | str] = ()
        match self.params.operation:
            case 'CAS':
                operands = rf'{PatternBuilder.predt()}, {PatternBuilder.regz()}, {{addr}}, {PatternBuilder.reg()}, {PatternBuilder.reg()}'
                dtype = (self.params.dtype[1],) if self.params.dtype is not None and self.params.dtype[1] > 32 else ()
            case _:
                operands = rf'{PatternBuilder.predt()}, {PatternBuilder.regz()}, {{addr}}, {PatternBuilder.reg()}'
                if self.params.dtype is not None and self.params.operation == 'EXCH':
                    dtype = (self.params.dtype[1],) if self.params.dtype[1] > 32 else ()
                elif self.params.dtype is not None:
                    match self.params.dtype[0]:
                        case 'F':
                            dtype = (
                                f'{self.params.dtype[0]}{self.params.dtype[1]}',
                                '?FTZ',
                                '?RN',
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

        match self.arch.compute_capability.as_int:
            case 70 | 75:
                addr = PatternBuilder.reg_addr()
            case 80 | 86 | 89:
                addr = PatternBuilder.either(PatternBuilder.reg_addr(), PatternBuilder.reg64_addr())
            case 90 | 100 | 120:
                addr = PatternBuilder.either(PatternBuilder.reg_addr(), PatternBuilder.desc_reg64_addr())
            case _:
                raise ValueError(f'unsupported {self.arch}')

        return PatternBuilder.opcode_mods(
            f'ATOM{self.params.memory}',
            ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope),
        ) + f' {operands.format(addr = addr)}'

class OpcodeModsMatcher(PatternMatcher):
    """
    Matcher that will collect all operands of an instruction.

    Useful when the opcode and modifiers are known and the operands may need to be retrieved.

    >>> from reprospect.test.sass import OpcodeModsMatcher
    >>> OpcodeModsMatcher(instruction = 'ISETP.NE.AND').matches(
    ...     'ISETP.NE.AND P2, PT, R4, RZ, PT'
    ... ).captures('operands')
    ['P2', 'PT', 'R4', 'RZ', 'PT']
    """
    @typeguard.typechecked
    def __init__(self, instruction : str, operands : bool = True) -> None:
        super().__init__(pattern = rf'^{instruction}\s+{PatternBuilder.operands()}' if operands else rf'^{instruction}')

class OpcodeModsWithOperandsMatcher(PatternMatcher):
    """
    Matcher that matches a given instruction and operands.

    Similar to :py:class:`OpcodeModsMatcher`, but the operands can be better constrained.

    >>> from reprospect.test.sass import OpcodeModsWithOperandsMatcher, PatternBuilder
    >>> OpcodeModsWithOperandsMatcher(
    ...     instruction = 'ISETP.NE.AND',
    ...     operands = (
    ...         PatternBuilder.PRED,
    ...         PatternBuilder.PREDT,
    ...         'R4',
    ...         PatternBuilder.REGZ,
    ...         PatternBuilder.PREDT,
    ...     )
    ... ).matches('ISETP.NE.AND P2, PT, R4, RZ, PT').captures('operands')
    ['P2', 'PT', 'R4', 'RZ', 'PT']
    """
    @typeguard.typechecked
    def __init__(self, instruction : str, operands : typing.Iterable[str]) -> None:
        operands = r',\s+'.join(
            PatternBuilder.group(op, group = 'operands') for op in operands
        )
        super().__init__(pattern = rf"^{instruction}\s+{operands}")
