"""
At first glance, examining generated SASS code may appear to be an esoteric task suited only to expert lab researchers — let alone testing it.

Yet, studying SASS — and assembly code in general — offers valuable insights.
Indeed, modern HPC code bases rely on complex software stacks and compiler toolchains.
While code correctness is often checked through regression testing,
reaching and sustaining optimal performance as software and hardware evolve requires additional effort.
This is usually achieved through verification of compile flags and *ad hoc* profiling and benchmarking.
However, beyond runtime analysis, the SASS code already contains information about the available code paths and could itself be incorporated into testing.
Still, the barrier to entry for meaningful SASS analysis is high: results can vary dramatically with compiler versions, optimization flags, and target architectures.

`ReProspect` provides a hierarchy of SASS instruction matchers that capture the components of an instruction (*opcodes*, *modifiers* and *operands*).
Under the hood, they generate complex regular expression (*regex*) patterns.
To accommodate for the evolving CUDA Instruction Set,
some of these matchers take the target architecture as a parameter and adjust the regex patterns accordingly.
In this way, `ReProspect` helps developers write assertions about expected instructions,
while reducing the need to track low-level details of the evolving CUDA instruction set.

.. doctest::

    >>> from reprospect.tools.architecture    import NVIDIAArch
    >>> from reprospect.test.sass.instruction import LoadGlobalMatcher
    >>> LoadGlobalMatcher(arch = NVIDIAArch.from_str('VOLTA70')).matches(inst = 'LDG.E.SYS R15, [R8+0x10]')
    InstructionMatch(opcode='LDG', modifiers=('E', 'SYS'), operands=('R15', 'R8+0x10'), predicate=None, additional={'address': ['R8+0x10']})
    >>> LoadGlobalMatcher(arch = NVIDIAArch.from_str('BLACKWELL120'), size = 128, readonly = True).matches(inst = 'LDG.E.128.CONSTANT R2, desc[UR15][R6.64+0x12]')
    InstructionMatch(opcode='LDG', modifiers=('E', '128', 'CONSTANT'), operands=('R2', 'desc[UR15][R6.64+0x12]'), predicate=None, additional={'address': ['desc[UR15][R6.64+0x12]']})

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

import mypy_extensions
import regex
import semantic_version

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass         import Instruction
from reprospect.tools.sass.decode  import RegisterType

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class PatternBuilder:
    """
    Helper class to build patterns for instruction components.
    """
    HEX : typing.Final[str] = r'0x[0-9A-Fa-f]+'
    OFFSET : typing.Final[str] = r'\+' + HEX
    PRED : typing.Final[str] = r'P[0-9]+'
    REG : typing.Final[str] = r'R[0-9]+'
    REG64 : typing.Final[str] = r'R[0-9]+\.64'
    UREG : typing.Final[str] = r'UR[0-9]+'

    #: Match a register or ``RZ``.
    REGZ : typing.Final[str] = r'R(?:Z|\d+)'

    #: Match a predicate register or ``PT``.
    PREDT : typing.Final[str] = r'P(?:T|\d+)'

    #: Match a uniform predicate register.
    UPRED : typing.Final[str] = r'UP[0-9]+'

    #: Match a uniform predicate register or ``UPT``.
    UPREDT : typing.Final[str] = r'UP(?:T|\d+)'

    OPERAND : typing.Final[str] = r'[\w@!\.\[\]\+\-\s]+'

    CONSTANT_BANK : typing.Final[str] = r'0x[0-9]+'
    """
    Constant memory bank.
    """

    CONSTANT_OFFSET : typing.Final[str] = r'(0x[0-9c]+|' + REG + '|' + UREG + ')'
    """
    Constant memory offset.
    """

    CONSTANT : typing.Final[str] = r'c\[' + CONSTANT_BANK + r'\]\[' + CONSTANT_OFFSET + r'\]'
    """
    Match constant memory location.
    The bank looks like ``0x3`` while the address is either compile-time (*e.g.*
    ``0x899``) or depends on a register.
    """

    IMMEDIATE : typing.Final[str] = '[0-9.]+'
    """
    References:

    * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L34
    """

    PREDICATE : typing.Final[str] = r'@!?U?P(?:T|\d+)'
    """
    Predicate for the whole instruction (comes before the opcode).
    """

    @staticmethod
    def zero_or_one(s : str) -> str:
        """
        Build an optional non-capturing pattern that matches zero or one occurrence of the given pattern.
        """
        return rf'(?:{s})?'

    @staticmethod
    def zero_or_more(s : str) -> str:
        """
        Build an optional non-capturing pattern that matches zero or more occurrences of the given pattern.
        """
        return rf'(?:{s})*'

    @staticmethod
    def any(*args : str) -> str:
        """
        Build a pattern matching any of `args`.
        """
        return f'({"|".join(args)})'

    @staticmethod
    def group(s : int | str, group : str) -> str:
        """
        Wrap a pattern in a named capture group.
        """
        return rf'(?P<{group}>{s})'

    @staticmethod
    def groups(s : int | str, groups : typing.Iterable[str]) -> str:
        """
        Wrap a pattern in named capture groups.
        """
        for g in groups:
            s = PatternBuilder.group(s, group = g)
        return str(s)

    @classmethod
    def reg(cls) -> str:
        """
        :py:attr:`REG` with `operands` group.
        """
        return cls.group(cls.REG, group = 'operands')

    @classmethod
    def mathreg(cls) -> str:
        """
        :py:attr:`REG` with `operands` group and optional math modifiers.

        References:

        * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L67
        """
        return cls.group(
            r'[\-]?' + cls.REG,
            group = 'operands',
        )

    @classmethod
    def regz(cls) -> str:
        """
        :py:attr:`REGZ` with `operands` group.
        """
        return cls.group(cls.REGZ, group = 'operands')

    @classmethod
    def ureg(cls) -> str:
        """
        :py:attr:`UREG` with `operands` group.
        """
        return cls.group(cls.UREG, group = 'operands')

    @classmethod
    def anygpreg(cls, reuse : typing.Optional[bool] = None, group : typing.Optional[str] = None) -> str:
        """
        Match any general purpose register.
        """
        pattern = cls.any(cls.REG, cls.UREG)
        if reuse is None:
            pattern += PatternBuilder.zero_or_one(r'\.reuse')
        elif reuse is True:
            pattern += r'\.reuse'
        if group is not None:
            pattern = cls.group(pattern, group = group)
        return pattern

    @classmethod
    def predt(cls) -> str:
        """
        :py:attr:`PREDT` with `operands` group.
        """
        return cls.group(cls.PREDT, group = 'operands')

    @classmethod
    def constant(cls) -> str:
        """
        :py:attr:`CONSTANT` with `operands` group.
        """
        return cls.group(cls.CONSTANT, group = 'operands')

    @classmethod
    def immediate(cls) -> str:
        """
        :py:attr:`IMMEDIATE` with `operands` group.
        """
        return cls.group(cls.IMMEDIATE, group = 'operands')

    @classmethod
    def predicate(cls) -> str:
        """
        :py:attr:`PREDICATE` with `predicate` group.
        """
        return cls.group(s = cls.PREDICATE, group = 'predicate')

    @staticmethod
    def opcode_mods(opcode : str, modifiers : typing.Optional[typing.Iterable[int | str | None]] = None) -> str:
        """
        Append each modifier with a `.`, within a proper named capture group.

        Note that the modifiers starting with a `?` are matched optionally.
        """
        opcode = PatternBuilder.group(opcode, group = 'opcode')
        if modifiers is not None:
            for modifier in filter(None, modifiers):
                modifier = typing.cast(int | str, modifier)
                if isinstance(modifier, str) and modifier.startswith('?'):
                    opcode += PatternBuilder.zero_or_one(r'\.' + PatternBuilder.group(modifier[1::], group = 'modifiers'))
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
    def address(cls, store : str, offset : typing.Optional[bool] = None) -> str:
        """
        Address operand. A few examples:

        * ``[R4]``
        * ``[R1.64]``
        * ``[R2+0x10]``

        :param store: The pattern storing the address, *e.g.* :py:attr:`REG`.
        :param offset: If not given, the offset matching is optional. It `True`, there must be an offset. If `False`, there must not be an offset.
        """
        if offset is None:
            store += cls.zero_or_one(cls.OFFSET)
        elif offset is True:
            store += cls.OFFSET
        return rf"\[{cls.groups(store, groups = ('operands', 'address'))}\]"

    @classmethod
    def desc_reg64_addr(cls, offset : typing.Optional[bool] = None) -> str:
        """
        Address operand with cache policy descriptor, like ``desc[UR0][R0.64+0x10]``.

        :param offset: If not given, the offset matching is optional. It `True`, there must be an offset. If `False`, there must not be an offset.
        """
        pattern = rf'desc\[({cls.UREG})\]'
        if offset is None:
            pattern += rf'\[({cls.REG64}{cls.zero_or_one(cls.OFFSET)})\]'
        elif offset is True:
            pattern += rf'\[({cls.REG64}{cls.OFFSET})\]'
        else:
            pattern += rf'\[({cls.REG64})\]'
        return cls.groups(pattern, groups = ('operands', 'address'))

def check_memory_instruction_word_size(*, size : int) -> None:
    """
    From https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses:

        Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes.
    """
    ALLOWABLE_SIZES : tuple[int, ...] = (1, 2, 4, 8, 16) # pylint: disable=invalid-name
    if size not in ALLOWABLE_SIZES:
        raise RuntimeError(f'{size} is not an allowable memory instruction word size ({ALLOWABLE_SIZES} (in bytes)).')

REGISTER_MATCH : typing.Final[regex.Pattern[str]] = regex.compile(
    r'^(?P<rtype>(R|UR|P|UP))'
    r'((?P<special>(T|Z))|(?P<index>\d+))?'
    r'(?P<reuse>\.reuse)?$'
)

@dataclasses.dataclass(frozen = True, slots = True)
class RegisterMatch:
    rtype : RegisterType
    index : int
    reuse : bool = False

    @classmethod
    def parse(cls, value : str) -> 'RegisterMatch':
        """
        Parse an operand, assuming it is a register.

        >>> from reprospect.test.sass.instruction import RegisterMatch
        >>> RegisterMatch.parse('UR12')
        RegisterMatch(rtype=<RegisterType.UGPR: 'UR'>, index=12, reuse=False)
        """
        if (matched := REGISTER_MATCH.match(value)) is not None:
            if matched['index'] or matched['special']:
                return cls(
                    rtype = RegisterType(matched['rtype']),
                    index = int(matched['index']) if matched['index'] else -1,
                    reuse = bool(matched.group('reuse')),
                )
        raise ValueError(f'Invalid register format {value!r}.')

@dataclasses.dataclass(frozen = True, slots = True)
class InstructionMatch:
    """
    An instruction with parsed components.
    """
    opcode : str
    modifiers : tuple[str, ...]
    operands : tuple[str, ...]
    predicate : str | None = None

    additional : dict[str, list[str]] | None = None

    @staticmethod
    def parse(*, bits : regex.Match[str]) -> 'InstructionMatch':
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
            opcode = opcode[0],
            modifiers = tuple(captured.pop('modifiers', ())),
            operands = tuple(captured.pop('operands', ())),
            predicate = predicate[0] if (predicate is not None and len(predicate) == 1) else None,
            additional = captured or None,
        )

@mypy_extensions.mypyc_attr(allow_interpreted_subclasses = True)
class InstructionMatcher(abc.ABC):
    """
    Abstract base class for instruction matchers.
    """
    @abc.abstractmethod
    def matches(self, inst : Instruction | str) -> typing.Optional[InstructionMatch]:
        """
        Check if the instruction matches.
        """

    def __call__(self, inst : Instruction | str) -> typing.Optional[InstructionMatch]:
        """
        Allow the matcher to be called as a function.
        """
        return self.matches(inst)

@mypy_extensions.mypyc_attr(allow_interpreted_subclasses = True)
class PatternMatcher(InstructionMatcher):
    """
    Regex-based (or pattern) matching.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('pattern',)

    def __init__(self, pattern : str | regex.Pattern[str]) -> None:
        self.pattern : typing.Final[str | regex.Pattern[str]] = pattern

    @override
    @typing.final
    def matches(self, inst : Instruction | str) -> typing.Optional[InstructionMatch]:
        if (matched := regex.match(self.pattern, inst.instruction if isinstance(inst, Instruction) else inst)) is not None:
            return InstructionMatch.parse(bits = matched)
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pattern={self.pattern})'

class FloatAddMatcher(PatternMatcher):
    """
    Matcher for floating-point add (``FADD``) instructions.
    """
    PATTERN : typing.Final[regex.Pattern[str]] = regex.compile(
        PatternBuilder.opcode_mods('FADD', modifiers = ('?FTZ',)) + r'\s+' +
        PatternBuilder.groups(PatternBuilder.REG, groups = ('dst', 'operands')) + r'\s*,\s*' +
        PatternBuilder.mathreg() + r'\s*,\s*' +
        PatternBuilder.any(
            PatternBuilder.mathreg(), PatternBuilder.ureg(),
            PatternBuilder.constant(),
            PatternBuilder.immediate(),
        )
    )

    def __init__(self) -> None:
        super().__init__(pattern = self.PATTERN)

class ArchitectureAwarePatternMatcher(PatternMatcher):
    """
    Base class for matchers that generate patterns based on architecture.
    """
    def __init__(self, arch : NVIDIAArch) -> None:
        self.arch = arch
        super().__init__(pattern = self._build_pattern())

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
    def __init__(self, arch : NVIDIAArch, version : typing.Optional[semantic_version.Version] = None) -> None:
        self.version = version if version is not None else semantic_version.Version(os.environ['CUDA_VERSION'])
        super().__init__(arch = arch)

class LoadGlobalMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for global load (``LDG``) instructions,
    like ``LDG.E R2, desc[UR6][R2.64]``.

    References:

    * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldg#global-memory-5-x
    * https://github.com/AdaptiveCpp/AdaptiveCpp/issues/848
    * https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsglobal.htm
    """
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
    def _build_pattern(self) -> str:
        match self.arch.compute_capability.as_int:
            case 70 | 75:
                return PatternBuilder.opcode_mods('LDG', ('E', self.params.size, self.params.cache, 'SYS')) + f' {PatternBuilder.reg()}, {PatternBuilder.address(PatternBuilder.REG)}'
            case 80 | 86 | 89:
                return PatternBuilder.opcode_mods('LDG', ('E', self.params.size, self.params.cache)) + f' {PatternBuilder.reg()}, {PatternBuilder.address(PatternBuilder.REG64)}'
            case 90 | 100 | 120:
                return PatternBuilder.opcode_mods('LDG', ('E', self.params.size, self.params.cache)) + f' {PatternBuilder.reg()}, {PatternBuilder.desc_reg64_addr()}'
            case _:
                raise ValueError(f'unsupported {self.arch}')

class LoadConstantMatcher(PatternMatcher):
    """
    Matcher for constant load (``LDC``) instructions, like:

    * ``LDC.64 R2, c[0x0][0x388]``
    * ``LDC R4, c[0x3][R0]``
    * ``LDCU UR4, c[0x3][UR0]``
    """
    CONSTANT : typing.Final[str] = PatternBuilder.group(
        r'c\['
        + PatternBuilder.group(PatternBuilder.CONSTANT_BANK, group = 'bank')
        + r'\]\['
        + PatternBuilder.group(PatternBuilder.CONSTANT_OFFSET, group = 'offset')
        + r'\]',
        group = 'operands',
    )

    def __init__(self, uniform : typing.Optional[bool] = None, size : typing.Optional[int] = None) -> None:
        """
        :param size: Optional bit size (e.g., 32, 64, 128).
        :param uniform: Optionally require uniformness.
        """
        if size is not None:
            check_memory_instruction_word_size(size = int(size / 8))

        if uniform is None:
            opcode = PatternBuilder.any('LDC', 'LDCU')
            dest   = PatternBuilder.anygpreg(reuse = False)
        elif uniform is True:
            opcode = 'LDCU'
            dest   = PatternBuilder.UREG
        else:
            opcode = 'LDC'
            dest   = PatternBuilder.REG

        pattern = PatternBuilder.opcode_mods(opcode, (size,)) + ' ' + PatternBuilder.group(dest, group = 'operands') + ', ' + self.CONSTANT

        super().__init__(pattern = pattern)

class StoreGlobalMatcher(ArchitectureAwarePatternMatcher):
    """
    Architecture-dependent matcher for global store (``STG``) instructions,
    like ``STG.E desc[UR6][R6.64], R15``.
    """
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
    def _build_pattern(self) -> str:
        match self.arch.compute_capability.as_int:
            case 70 | 75:
                return PatternBuilder.opcode_mods('STG', ('E', self.params.size, 'SYS')) + f' {PatternBuilder.address(PatternBuilder.REG)}, {PatternBuilder.reg()}'
            case 80 | 86 | 89:
                return PatternBuilder.opcode_mods('STG', ('E', self.params.size)) + f' {PatternBuilder.address(PatternBuilder.REG64)}, {PatternBuilder.reg()}'
            case 90 | 100 | 120:
                return PatternBuilder.opcode_mods('STG', ('E', self.params.size)) + f' {PatternBuilder.desc_reg64_addr()}, {PatternBuilder.reg()}'
            case _:
                raise ValueError(f'unsupported {self.arch}')

ThreadScope = typing.Literal['BLOCK', 'DEVICE', 'THREADS']
"""
References:

* https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes
"""

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
        dtype : typing.Sequence[int | str]
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
        else:
            dtype = ()

        match self.arch.compute_capability.as_int:
            case 70 | 75:
                return PatternBuilder.opcode_mods('RED', ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)) + f' {PatternBuilder.address(PatternBuilder.REG)}, {PatternBuilder.reg()}'
            case 80 | 86 | 89:
                return PatternBuilder.opcode_mods('RED', ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)) + f' {PatternBuilder.address(PatternBuilder.REG64)}, {PatternBuilder.reg()}'
            case 90 | 100 | 120:
                return PatternBuilder.opcode_mods('REDG', ('E', self.params.operation, *dtype, self.params.consistency, self.params.scope)) + f' {PatternBuilder.desc_reg64_addr()}, {PatternBuilder.reg()}'
            case _:
                raise ValueError(f'unsupported {self.arch}')

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
        super().__init__(arch = arch, version = version)

    @override
    def _build_pattern(self) -> str: # pylint: disable=too-many-branches
        """
        ``CAS`` has a different operand structure. For instance::

            ATOMG.E.CAS.STRONG.GPU PT, R7, [R2], R6, R7

        The generic pattern for the operands is thus::

            {pred}, {dest}, [{addr}], {compare}, {newval}
        """
        dtype : typing.Sequence[int | str]
        match self.params.operation:
            case 'CAS':
                operands = rf'{PatternBuilder.predt()}, {PatternBuilder.regz()}, {{addr}}, {PatternBuilder.reg()}, {PatternBuilder.reg()}'
                dtype = (self.params.dtype[1],) if self.params.dtype is not None and self.params.dtype[1] > 32 else ()
            case _:
                operands = rf'{PatternBuilder.predt()}, {PatternBuilder.regz()}, {{addr}}, {PatternBuilder.regz()}'
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
                else:
                    dtype = ()

        match self.arch.compute_capability.as_int:
            case 70 | 75:
                addr = PatternBuilder.any(PatternBuilder.address(PatternBuilder.REG), PatternBuilder.address(PatternBuilder.UREG))
            case 80 | 86 | 89:
                addr = PatternBuilder.any(PatternBuilder.address(PatternBuilder.REG), PatternBuilder.address(PatternBuilder.REG64))
            case 90 | 100 | 120:
                addr = PatternBuilder.any(PatternBuilder.address(PatternBuilder.REG), PatternBuilder.desc_reg64_addr())
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

    >>> from reprospect.test.sass.instruction import OpcodeModsMatcher
    >>> OpcodeModsMatcher(opcode = 'ISETP', modifiers = ('NE', 'AND')).matches(
    ...     'ISETP.NE.AND P2, PT, R4, RZ, PT'
    ... )
    InstructionMatch(opcode='ISETP', modifiers=('NE', 'AND'), operands=('P2', 'PT', 'R4', 'RZ', 'PT'), predicate=None, additional=None)
    """
    def __init__(self, *,
        opcode : str,
        modifiers : typing.Optional[typing.Iterable[str]] = None,
        operands : bool = True,
    ) -> None:
        pattern = '^' + PatternBuilder.opcode_mods(opcode, modifiers)
        if operands:
            pattern += rf'\s+{PatternBuilder.operands()}'
        super().__init__(pattern = pattern)

class OpcodeModsWithOperandsMatcher(PatternMatcher):
    """
    Matcher that matches a given instruction and operands.

    Similar to :py:class:`OpcodeModsMatcher`, but the operands can be better constrained.

    >>> from reprospect.test.sass.instruction import OpcodeModsWithOperandsMatcher, PatternBuilder
    >>> OpcodeModsWithOperandsMatcher(
    ...     opcode = 'ISETP',
    ...     modifiers = ('NE', 'AND'),
    ...     operands = (
    ...         PatternBuilder.PRED,
    ...         PatternBuilder.PREDT,
    ...         'R4',
    ...         PatternBuilder.REGZ,
    ...         PatternBuilder.PREDT,
    ...     )
    ... ).matches('ISETP.NE.AND P2, PT, R4, RZ, PT')
    InstructionMatch(opcode='ISETP', modifiers=('NE', 'AND'), operands=('P2', 'PT', 'R4', 'RZ', 'PT'), predicate=None, additional=None)

    .. note::

        Some operands can be optionally matched.

        >>> from reprospect.test.sass.instruction import OpcodeModsWithOperandsMatcher, PatternBuilder
        >>> matcher = OpcodeModsWithOperandsMatcher(opcode = 'WHATEVER', operands = (
        ...     PatternBuilder.zero_or_one('R0'),
        ...     PatternBuilder.zero_or_one('R9'),
        ... ))
        >>> matcher.matches('WHATEVER')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('',), predicate=None, additional=None)
        >>> matcher.matches('WHATEVER R0')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('R0',), predicate=None, additional=None)
        >>> matcher.matches('WHATEVER R0, R9')
        InstructionMatch(opcode='WHATEVER', modifiers=(), operands=('R0', 'R9'), predicate=None, additional=None)
    """
    SEPARATOR : typing.Final[str] = r',\s+'

    @classmethod
    def operand(cls, *, op : str) -> str:
        pattern = PatternBuilder.group(op, group = 'operands')
        if op.startswith('(') and op.endswith(')?'):
            return PatternBuilder.zero_or_more(cls.SEPARATOR + pattern)
        return cls.SEPARATOR + pattern

    def __init__(self, *,
        opcode : str,
        operands : typing.Iterable[str],
        modifiers : typing.Optional[typing.Iterable[str]] = None,
    ) -> None:
        ops_iter = iter(operands)
        ops : str = PatternBuilder.group(next(ops_iter), group = 'operands')
        ops += ''.join(self.operand(op = op) for op in ops_iter)
        super().__init__(pattern = rf'^{PatternBuilder.opcode_mods(opcode, modifiers)}\s*{ops}')

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
    >>> AnyMatcher().matches(inst = 'FADD.FTZ.RN R0, R1, R2')
    InstructionMatch(opcode='FADD', modifiers=('FTZ', 'RN'), operands=('R0', 'R1', 'R2'), predicate=None, additional=None)
    >>> AnyMatcher().matches(inst = 'RET.REL.NODEC R4 0x0')
    InstructionMatch(opcode='RET', modifiers=('REL', 'NODEC'), operands=('R4', '0x0'), predicate=None, additional=None)
    """
    PATTERN : typing.Final[regex.Pattern[str]] = regex.compile(
        r'^'
        + PatternBuilder.zero_or_one(PatternBuilder.predicate() + r'\s*')
        + PatternBuilder.group(s = r'[A-Z0-9]+', group = 'opcode')
        + PatternBuilder.zero_or_more(s = r'\.' + PatternBuilder.group(s = r'[A-Z0-9_]+', group = 'modifiers'))
        + r'\s*'
        + PatternBuilder.zero_or_more(s = PatternBuilder.group(s = r'[^,\s]+', group = 'operands') + PatternBuilder.any(r'\s*,\s*', r'\s+'))
        + PatternBuilder.zero_or_one(s = PatternBuilder.group(s = r'[^,\s]+', group = 'operands'))
        + r'$'
    )

    def __init__(self):
        super().__init__(pattern = self.PATTERN)

class BranchMatcher(PatternMatcher):
    """
    Matcher for a ``BRA`` branch instruction.

    Typically::

        @!UP5 BRA 0x456
    """
    BRA : typing.Final[str] = (
        PatternBuilder.opcode_mods(opcode = 'BRA')
        + r'\s*'
        + PatternBuilder.group(PatternBuilder.HEX, group = 'operands')
        + r'$'
    )

    INSTRUCTION_PREDICATE : typing.Final[str] = PatternBuilder.zero_or_one(PatternBuilder.predicate() + r'\s*')

    def __init__(self, predicate : typing.Optional[str] = None):
        super().__init__(pattern = r'^' + (self.INSTRUCTION_PREDICATE if predicate is None else PatternBuilder.group(predicate, group = 'predicate') + r'\s+') + self.BRA)
