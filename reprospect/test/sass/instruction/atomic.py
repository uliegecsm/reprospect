import sys
import typing

import semantic_version

from reprospect.test.sass.instruction.address import AddressMatcher
from reprospect.test.sass.instruction.instruction import (
    ArchitectureAndVersionAwarePatternMatcher,
    ArchitectureAwarePatternMatcher,
    OpCode,
    ZeroOrOne,
)
from reprospect.test.sass.instruction.memory import (
    MemorySpace,
    check_memory_instruction_word_size,
)
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch
from reprospect.utils.types import ConvertibleTypeInfo, Kind, TypeInfo

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class ThreadScope(StrEnum):
    """
    References:

    * https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes
    """
    BLOCK = 'BLOCK'
    DEVICE = 'DEVICE'
    SYSTEM = 'SYSTEM'
    THREADS = 'THREADS'

    def convert(self, *, arch: NVIDIAArch) -> str:
        """
        Convert to SASS modifier.

        For instance, ``__NV_THREAD_SCOPE_DEVICE`` maps to ``GPU``.

        References:

        * https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions

        .. warning::

            ``__NV_THREAD_SCOPE_THREAD`` thread scope is currently implemented using wider ``__NV_THREAD_SCOPE_BLOCK`` thread scope.
        """
        match self:
            case self.THREADS | self.BLOCK:
                if arch.compute_capability < 80:
                    return 'CTA'
                return 'SM'
            case self.DEVICE:
                return 'GPU'
            case self.SYSTEM:
                return 'SYS'
            case _:
                raise ValueError(self)

class ReductionMatcher(ArchitectureAwarePatternMatcher):
    """
    Matcher for reduction operations on generic memory (``RED``).

    ``RED`` instructions are typically used when the atomic operation return value is not used.
    Otherwise, it would typically map to ``ATOM``.

    The ``RED`` opcode may take several modifiers:

    * operation (*e.g.* `ADD`)
    * scope (*e.g.* :py:attr:`reprospect.test.sass.instruction.atomic.ThreadScope.DEVICE`)
    * consistency (*e.g.* `STRONG`)

    References:

    * https://forums.developer.nvidia.com/t/difference-between-red-and-atomg-sass-instruction/203469
    """
    __slots__ = ('consistency', 'dtype', 'operation', 'scope')

    TEMPLATE: typing.Final[str] = f'{{opcode}} {{address}}, {Register.reg()}'

    def __init__(self,
        arch: NVIDIAArch,
        operation: str = 'ADD',
        scope: ThreadScope | str | None = None,
        consistency: str = 'STRONG',
        dtype: ConvertibleTypeInfo | None = None,
    ) -> None:
        self.operation: typing.Final[str] = operation
        self.scope: typing.Final[str | None] = ThreadScope(scope).convert(arch=arch) if scope else None
        self.consistency: typing.Final[str] = consistency
        self.dtype: typing.Final[TypeInfo | None] = TypeInfo.normalize(dtype=dtype) if dtype else None

        if self.dtype is not None:
            check_memory_instruction_word_size(size=self.dtype.itemsize)

        super().__init__(arch=arch)

    @override
    def _build_pattern(self) -> str:
        dtype: typing.Sequence[int | str | ZeroOrOne]
        if self.dtype is not None:
            match self.dtype.kind:
                case Kind.FLOAT:
                    dtype = (
                        f'F{self.dtype.bits}',
                        ZeroOrOne('FTZ'),
                        ZeroOrOne('RN'),
                    )
                case Kind.INT:
                    if self.operation == 'ADD':
                        dtype = ()
                    else:
                        dtype = (f'S{self.dtype.bits}',)
                case Kind.UINT:
                    dtype = (self.dtype.bits,) if self.dtype.bits > 32 else ()
                case _:
                    raise ValueError(self.dtype)
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
                modifiers=filter(None, ('E', self.operation, *dtype, self.consistency, self.scope)),
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
    * scope (*e.g.* :py:attr:`reprospect.test.sass.instruction.atomic.ThreadScope.DEVICE`)
    * consistency (*e.g.* `STRONG`)

    References:

    * https://docs.nvidia.com/cuda/archive/12.6.3/cuda-binary-utilities/index.html#hopper-instruction-set
    """
    __slots__ = ('consistency', 'dtype', 'memory', 'operation', 'scope')

    TEMPLATE_CAS: typing.Final[str] = rf'{{opcode}} {Register.predt()}, {Register.regz()}, {{address}}, {Register.reg()}, {Register.reg()}'
    TEMPLATE: typing.Final[str]     = rf'{{opcode}} {Register.predt()}, {Register.regz()}, {{address}}, {Register.regz()}'

    def __init__(self,
        arch: NVIDIAArch,
        operation: str = 'ADD',
        scope: ThreadScope | str | None = None,
        consistency: str = 'STRONG',
        dtype: ConvertibleTypeInfo | None = None,
        memory: MemorySpace | str = MemorySpace.GLOBAL,
        version: semantic_version.Version | None = None,
    ) -> None:
        self.operation: typing.Final[str] = operation
        self.scope: typing.Final[str | None] = ThreadScope(scope).convert(arch=arch) if scope else None
        self.consistency: typing.Final[str] = consistency
        self.memory: typing.Final[MemorySpace] = MemorySpace(memory)
        self.dtype: typing.Final[TypeInfo | None] = TypeInfo.normalize(dtype=dtype) if dtype else None

        if self.dtype is not None:
            check_memory_instruction_word_size(size=self.dtype.itemsize)

        super().__init__(arch=arch, version=version)

    @override
    def _build_pattern(self) -> str:
        """
        ``CAS`` has a different operand structure. For instance::

            ATOMG.E.CAS.STRONG.GPU PT, R7, [R2], R6, R7

        The generic pattern for the operands is thus::

            {pred}, {dest}, [{addr}], {compare}, {newval}
        """
        dtype: typing.Sequence[int | str | ZeroOrOne]
        match self.operation:
            case 'CAS' | 'EXCH':
                dtype = (self.dtype.bits,) if self.dtype is not None and self.dtype.bits > 32 else ()
            case _:
                if self.dtype is not None:
                    match self.dtype.kind:
                        case Kind.FLOAT:
                            dtype = (
                                f'F{self.dtype.bits}',
                                ZeroOrOne('FTZ'),
                                ZeroOrOne('RN'),
                            )
                        case Kind.INT:
                            if self.operation == 'ADD' and self.version in semantic_version.SimpleSpec('<12.8'):
                                dtype = ()
                            else:
                                dtype = (f'S{self.dtype.bits}',)
                        case Kind.UINT:
                            dtype = (self.dtype.bits,) if self.dtype.bits > 32 else ()
                        case _:
                            raise ValueError(self.dtype)
                else:
                    dtype = ()

        address: str = AddressMatcher.build_pattern(arch=self.arch, memory=self.memory)

        match self.arch.compute_capability.as_int:
            case 80 | 86 | 89 | 90 | 100 | 103 | 120:
                address = PatternBuilder.any(AddressMatcher.build_reg_address(arch=self.arch), address)
            case _:
                pass

        return (self.TEMPLATE_CAS if self.operation == 'CAS' else self.TEMPLATE).format(
            opcode=OpCode.mod(
                opcode=f'ATOM{self.memory}',
                modifiers=filter(None, ('E', self.operation, *dtype, self.consistency, self.scope)),
            ),
            address=PatternBuilder.groups(address, groups=('operands', 'address')),
        )
