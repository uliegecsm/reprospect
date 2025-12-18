"""
:code:`Kokkos` provides extended atomic support for objects of arbitrary size.
Therefore, it has to support types that are not directly handled by the backend.
This is achieved through the `desul <https://github.com/desul/desul>`_ library :cite:`ctrott-2022`,
that, depending on the size of the object and the targeted hardware, maps atomic operations to either:

1. atomic instruction
2. CAS loop
3. sharded lock table

Traditionally, CUDA atomics supported up to 64-bit size operations.
Since compute capability 9.0, CUDA `supports atomic CAS for objects up to 128-bit size <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomiccas>`_.
Therefore, there has been some effort in :code:`Kokkos` to bring this support through `desul`.
For instance, a :code:`Kokkos::atomic_add` for 128-bit aligned :code:`Kokkos::complex<double>` should use
the sharded lock table implementation only for compute capability below 9.0,
and resort to the CAS-based implementation otherwise.

- https://github.com/kokkos/kokkos/pull/8025
- https://github.com/kokkos/kokkos/pull/8495
- https://github.com/kokkos/kokkos/pull/8511

To ensure that :code:`Kokkos` implements the right code path, the following matchers can be used:

* :py:class:`examples.kokkos.atomic.cas.AtomicCAS`
* :py:class:`examples.kokkos.atomic.desul.LockBasedAtomicMatcher`

The following tests:

* :py:meth:`examples.kokkos.atomic.example_add_complex64.TestAtomicAddComplex64`
* :py:meth:`examples.kokkos.atomic.example_add_complex128.TestAtomicAddComplex128`
* :py:meth:`examples.kokkos.atomic.example_add_double256.TestAtomicAddDouble256`
* :py:meth:`examples.kokkos.atomic.example_add_int128.TestAtomicAddInt128`

verify that :code:`Kokkos::atomic_add` maps to the right implementation
by looking for an instruction sequence pattern.
"""

import logging
import sys
import typing

from reprospect.test.sass.composite import (
    any_of,
    instruction_is,
    instructions_are,
    instructions_contain,
)
from reprospect.test.sass.composite_impl import (
    InSequenceAtMatcher,
    OrderedInSequenceMatcher,
    SequenceMatcher,
)
from reprospect.test.sass.instruction import (
    AtomicMatcher,
    BranchMatcher,
    InstructionMatch,
    InstructionMatcher,
    LoadGlobalMatcher,
    OpcodeModsMatcher,
    OpcodeModsWithOperandsMatcher,
    PatternBuilder,
    StoreGlobalMatcher,
)
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

def get_atomic_memory_suffix(compiler_id: str) -> typing.Literal['G', '']:
    """
    See :py:meth:`tests.test.sass.test_atomic.TestAtomicMatcher.test_exch_device_ptr`.
    """
    match compiler_id:
        case 'NVIDIA':
            return 'G'
        case 'Clang':
            return ''
        case _:
            raise ValueError(f'unsupported compiler ID {compiler_id}')

class AtomicAcquireMatcher:
    """
    Matcher for the trial to acquire a lock through an atomic exchange.

    See:

    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L39
    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Array_CUDA.hpp#L83
    """
    @classmethod
    def build(cls, arch: NVIDIAArch, compiler_id: str) -> OrderedInSequenceMatcher:
        return instructions_are(
            # Storing 1 in a register can be done in different ways.
            any_of(
                OpcodeModsWithOperandsMatcher(
                    opcode='MOV',
                    operands=(PatternBuilder.REG, '0x1'),
                ),
                OpcodeModsWithOperandsMatcher(
                    opcode='IMAD', modifiers=('MOV', 'U32'),
                    operands=(PatternBuilder.REG, PatternBuilder.REGZ, PatternBuilder.REGZ, '0x1'),
                ),
            ),
            AtomicMatcher(
                memory=get_atomic_memory_suffix(compiler_id=compiler_id),
                arch=arch,
                operation='EXCH',
                scope='DEVICE',
                consistency='STRONG',
            ),
        )

class AtomicReleaseMatcher:
    """
    Matcher for the release of a lock through an atomic exchange.

    See:

    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L44
    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Array_CUDA.hpp#L102
    """
    @classmethod
    def build(cls, arch: NVIDIAArch, compiler_id: str) -> InstructionMatcher:
        return instruction_is(AtomicMatcher(
            memory=get_atomic_memory_suffix(compiler_id=compiler_id),
            arch=arch,
            operation='EXCH',
            scope='DEVICE',
            consistency='STRONG',
        )).with_operand(index=-1, operand='RZ')

class DeviceAtomicThreadFenceMatcher:
    """
    Matcher for the device atomic thread fence block.

    See:

    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L40
    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Thread_Fence_CUDA.hpp#L19

    """
    @classmethod
    def build(cls, arch: NVIDIAArch) -> OrderedInSequenceMatcher:
        matchers = [
            OpcodeModsMatcher(opcode='MEMBAR', modifiers=('SC', 'GPU'), operands=False),
            OpcodeModsMatcher(opcode='ERRBAR', operands=False),
        ]
        if arch.compute_capability >= 90:
            matchers.append(OpcodeModsMatcher(opcode='CGAERRBAR', operands=False))
        matchers.append(OpcodeModsMatcher(opcode='CCTL', modifiers=('IVALL',), operands=False))
        return instructions_are(*matchers)

class Operation(typing.Protocol):
    def build(self, loads: typing.Collection[InstructionMatch]) -> SequenceMatcher:
        ...

class LockBasedAtomicMatcher(SequenceMatcher):
    """"
    Matcher for the `desul` lock-based atomic code path.

    See:

    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L39-L44
    """
    def __init__(self, *,
        arch: NVIDIAArch,
        operation: Operation,
        compiler_id: str,
        size: int = 128,
        level: int = logging.INFO,
        load: SequenceMatcher | None = None,
        store: SequenceMatcher | None = None,
    ) -> None:
        self.arch: typing.Final[NVIDIAArch] = arch
        self.operation: typing.Final[Operation] = operation
        self.compiler_id: typing.Final[str] = compiler_id
        self.level: typing.Final[int] = level
        self.load:  typing.Final[SequenceMatcher] = load  or InSequenceAtMatcher(matcher=LoadGlobalMatcher (arch=self.arch, size=size, readonly=False))
        self.store: typing.Final[SequenceMatcher] = store or InSequenceAtMatcher(matcher=StoreGlobalMatcher(arch=self.arch, size=size))
        self._index: int = 0

    def collect(self, matched: list[InstructionMatch], new: InstructionMatch | list[InstructionMatch]) -> int:
        if isinstance(new, list):
            for elem in new:
                logging.log(self.level, elem)
            matched.extend(new)
            return len(new)

        logging.log(self.level, new)
        matched.append(new)
        return 1

    @override
    def match(self, instructions: typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None: # pylint: disable=too-many-branches,too-many-return-statements,too-many-statements
        """
        .. note::

            For data types that require many loads or stores, the operation instructions might be interleaved, such that the
            sequence within the memory thread fences is not strictly load/operation/store.
        """
        matched: list[InstructionMatch] = []

        # First, try to atomically acquire a lock.
        matcher_start = instructions_contain(matcher=AtomicAcquireMatcher.build(arch=self.arch, compiler_id=self.compiler_id))
        matched_atomic_acquire = matcher_start.match(instructions=instructions)

        if matched_atomic_acquire is None:
            return None
        self.collect(matched=matched, new=matched_atomic_acquire)
        offset = matcher_start.next_index

        # Then, one or two PLOP3.LUT instructions.
        matched_plop3_lut = instruction_is(
            OpcodeModsMatcher(opcode='PLOP3', modifiers=('LUT',)),
        ).one_or_more_times().match(instructions[offset:])

        if matched_plop3_lut is None:
            return None

        offset += self.collect(matched=matched, new=matched_plop3_lut)

        # Then, ISETP.NE.AND that reuses the register in which the atomic acquire put its result.
        modifiers: tuple[str, ...]
        match self.compiler_id:
            case 'NVIDIA':
                modifiers = ('NE', 'AND')
            case 'Clang':
                modifiers = ('NE', 'U32', 'AND')
            case _:
                raise ValueError(f'unsupported compiler ID {self.compiler_id}')
        matcher_isetp_enter = OpcodeModsWithOperandsMatcher(
            opcode='ISETP', modifiers=modifiers,
            operands=(
                PatternBuilder.PRED,
                PatternBuilder.PREDT,
                matched_atomic_acquire[-1].operands[1],
                PatternBuilder.REGZ,
                PatternBuilder.PREDT,
            ),
        )

        matched_isetp_enter = matcher_isetp_enter.match(instructions[offset])

        if matched_isetp_enter is None:
            return None

        offset += self.collect(matched=matched, new=matched_isetp_enter)

        # Next, it enters the branching. It looks like:
        #   @P[0-9]+ BRA offset
        outer_branch_p = matched_isetp_enter.operands[0]
        matcher_outer_branch = BranchMatcher(predicate=rf'@{outer_branch_p}')
        matched_outer_branch = matcher_outer_branch.match(instructions[offset])

        if matched_outer_branch is None:
            return None

        offset += self.collect(matched=matched, new=matched_outer_branch)

        # The device atomic thread fence block.
        matched_thread_fence = DeviceAtomicThreadFenceMatcher.build(arch=self.arch).match(instructions=instructions[offset:])

        if matched_thread_fence is None:
            return None

        offset += self.collect(matched=matched, new=matched_thread_fence)

        offset_fence_enter = offset

        # The device atomic thread fence block (again).
        matcher_thread_fence = instructions_contain(DeviceAtomicThreadFenceMatcher.build(arch=self.arch))
        matched_thread_fence = matcher_thread_fence.match(instructions=instructions[offset:])

        if matched_thread_fence is None:
            return None

        offset_fence_exit = offset + matcher_thread_fence.next_index - len(matched_thread_fence)
        offset += matcher_thread_fence.next_index
        self.collect(matched=matched, new=matched_thread_fence)

        # The load corresponds to the dereferencing line at
        # https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L41.
        matched_load = instructions_contain(self.load).match(instructions=instructions[offset_fence_enter:offset_fence_exit])

        if matched_load is None:
            return None

        self.collect(matched=matched, new=matched_load)

        # Operation.
        matcher_operation = instructions_contain(matcher=self.operation.build(loads=matched_load))
        matched_operation = matcher_operation.match(instructions=instructions[offset_fence_enter:offset_fence_exit])

        if matched_operation is None:
            return None

        self.collect(matched=matched, new=matched_operation)

        # Store the value.
        matched_store = instructions_contain(self.store).match(instructions=instructions[offset_fence_enter:offset_fence_exit])

        if matched_store is None:
            return None

        self.collect(matched=matched, new=matched_store)

        # Loads and stores use the same registers.
        assert len(matched_store) == len(matched_load)
        assert all(len(x.operands) == len(y.operands) for x, y in zip(matched_store, matched_load, strict=True))
        match len(matched_store[0].operands):
            case 2:
                if matched_store[0].operands[1] != matched_load[0].operands[0]:
                    return None
            case 3:
                if matched_store[0].operands[1] != matched_load[0].operands[1]:
                    return None
                if matched_store[0].operands[2] != matched_load[0].operands[0]:
                    return None
            case _:
                raise ValueError

        # Atomic release.
        matched_atomic_release = AtomicReleaseMatcher.build(arch=self.arch, compiler_id=self.compiler_id).match(inst=instructions[offset])

        if matched_atomic_release is None:
            return None

        self.collect(matched=matched, new=matched_atomic_release)

        self._index = offset + 1

        return matched

    @override
    @property
    def next_index(self) -> int:
        return self._index
