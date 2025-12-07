import logging
import sys
import typing

from reprospect.test.sass.composite      import any_of, \
                                                instructions_are, \
                                                instruction_is, \
                                                instructions_contain
from reprospect.test.sass.composite_impl import OrderedInSequenceMatcher, SequenceMatcher
from reprospect.test.sass.instruction    import AtomicMatcher, \
                                                BranchMatcher, \
                                                InstructionMatch,\
                                                InstructionMatcher,\
                                                LoadGlobalMatcher, \
                                                OpcodeModsMatcher, \
                                                OpcodeModsWithOperandsMatcher, \
                                                PatternBuilder, \
                                                StoreGlobalMatcher
from reprospect.tools.architecture       import NVIDIAArch
from reprospect.tools.sass               import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

def get_atomic_memory_suffix(compiler_id : str) -> typing.Literal['G', '']:
    """
    See :py:meth:`tests.python.test.sass.test_atomic.TestAtomicMatcher.test_exch_device_ptr`.
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
    def build(cls, arch : NVIDIAArch, compiler_id : str) -> OrderedInSequenceMatcher:
        return instructions_are(
            # Storing 1 in a register can be done in different ways.
            any_of(
                OpcodeModsWithOperandsMatcher(
                    opcode = 'MOV',
                    operands = (PatternBuilder.REG, '0x1'),
                ),
                OpcodeModsWithOperandsMatcher(
                    opcode = 'IMAD', modifiers = ('MOV', 'U32'),
                    operands = (PatternBuilder.REG, PatternBuilder.REGZ, PatternBuilder.REGZ, '0x1'),
                ),
            ),
            AtomicMatcher(
                memory = get_atomic_memory_suffix(compiler_id = compiler_id),
                arch = arch,
                operation = 'EXCH',
                scope = 'DEVICE',
                consistency = 'STRONG',
            ),
        )

class RegisterValidator(InstructionMatcher):
    """
    Matches if the underlying :py:attr:`matcher` matches **and** the last operand
    of the matched instruction is equal to :py:attr:`expected`.
    """
    def __init__(self, matcher : InstructionMatcher, expected : str) -> None:
        self.matcher : typing.Final[InstructionMatcher] = matcher
        self.expected : typing.Final[str] = expected

    @override
    def match(self, inst : Instruction | str) -> InstructionMatch | None:
        if (matched := self.matcher.match(inst = inst)) is not None:
            if matched.operands[-1] == self.expected:
                return matched
        return None

class AtomicReleaseMatcher:
    """
    Matcher for the release of a lock through an atomic exchange.

    See:

    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L44
    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Array_CUDA.hpp#L102
    """
    @classmethod
    def build(cls, arch : NVIDIAArch, compiler_id : str) -> InstructionMatcher:
        return instruction_is(RegisterValidator(AtomicMatcher(
            memory = get_atomic_memory_suffix(compiler_id = compiler_id),
            arch = arch,
            operation = 'EXCH',
            scope = 'DEVICE',
            consistency = 'STRONG',
        ), 'RZ'))

class DeviceAtomicThreadFenceMatcher:
    """
    Matcher for the device atomic thread fence block.

    See:

    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L40
    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Thread_Fence_CUDA.hpp#L19

    """
    @classmethod
    def build(cls, arch : NVIDIAArch) -> OrderedInSequenceMatcher:
        matchers = [
            OpcodeModsMatcher(opcode = 'MEMBAR', modifiers = ('SC', 'GPU'), operands = False),
            OpcodeModsMatcher(opcode = 'ERRBAR', operands = False),
        ]
        if arch.compute_capability >= 90:
            matchers.append(OpcodeModsMatcher(opcode = 'CGAERRBAR', operands = False))
        matchers.append(OpcodeModsMatcher(opcode = 'CCTL', modifiers = ('IVALL',), operands = False))
        return instructions_are(*matchers)

class Operation(typing.Protocol):
    def build(self, load : InstructionMatch) -> SequenceMatcher:
        ...

class LockBasedAtomicMatcher(SequenceMatcher):
    """"
    Matcher for the `desul` lock-based atomic code path.

    See:

    * https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L39-L44
    """
    def __init__(self, *,
        arch : NVIDIAArch,
        operation : Operation,
        compiler_id : str,
        size : int = 128,
        level : int = logging.INFO,
    ) -> None:
        self.arch : typing.Final[NVIDIAArch] = arch
        self.operation : typing.Final[Operation] = operation
        self.compiler_id : typing.Final[str] = compiler_id
        self.size = size
        self.level : typing.Final[int] = level

    def collect(self, matched : list[InstructionMatch], new : InstructionMatch | list[InstructionMatch]) -> int:
        if isinstance(new, list):
            for elem in new:
                logging.log(self.level, elem)
            matched.extend(new)
            return len(new)

        logging.log(self.level, new)
        matched.append(new)
        return 1

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None: # pylint: disable=too-many-branches,too-many-return-statements
        matched : list[InstructionMatch] = []

        # First, try to atomically acquire a lock.
        matcher_start = instructions_contain(matcher = AtomicAcquireMatcher.build(arch = self.arch, compiler_id = self.compiler_id))
        matched_atomic_acquire = matcher_start.match(instructions = instructions)

        if matched_atomic_acquire is None or matcher_start.index is None:
            return None

        offset = matcher_start.index + self.collect(matched = matched, new = matched_atomic_acquire)

        # Then, one or two PLOP3.LUT instructions.
        matched_plop3_lut = instruction_is(
            OpcodeModsMatcher(opcode = 'PLOP3', modifiers = ('LUT',)),
        ).one_or_more_times().match(instructions[offset::])

        if matched_plop3_lut is None:
            return None

        offset += self.collect(matched = matched, new = matched_plop3_lut)

        # Then, ISETP.NE.AND that reuses the register in which the atomic acquire put its result.
        modifiers : tuple[str, ...]
        match self.compiler_id:
            case 'NVIDIA':
                modifiers = ('NE', 'AND')
            case 'Clang':
                modifiers = ('NE', 'U32', 'AND')
            case _:
                raise ValueError(f'unsupported compiler ID {self.compiler_id}')
        matcher_isetp_enter = OpcodeModsWithOperandsMatcher(
            opcode = 'ISETP', modifiers = modifiers,
            operands = (
                PatternBuilder.PRED,
                PatternBuilder.PREDT,
                matched_atomic_acquire[-1].operands[1],
                PatternBuilder.REGZ,
                PatternBuilder.PREDT,
            )
        )

        matched_isetp_enter = matcher_isetp_enter.match(instructions[offset])

        if matched_isetp_enter is None:
            return None

        offset += self.collect(matched = matched, new = matched_isetp_enter)

        # Next, it enters the branching. It looks like:
        #   @P[0-9]+ BRA offset
        outer_branch_p = matched_isetp_enter.operands[0]
        matcher_outer_branch = BranchMatcher(predicate = rf'@{outer_branch_p}')
        matched_outer_branch = matcher_outer_branch.match(instructions[offset])

        if matched_outer_branch is None:
            return None

        offset += self.collect(matched = matched, new = matched_outer_branch)

        # The device atomic thread fence block.
        matched_thread_fence = DeviceAtomicThreadFenceMatcher.build(arch = self.arch).match(instructions = instructions[offset::])

        if matched_thread_fence is None:
            return None

        offset += self.collect(matched = matched, new = matched_thread_fence)

        # The load corresponds to the dereferencing line at
        # https://github.com/desul/desul/blob/79f928075837ffb5d302aae188e0ec7b7a79ae94/atomics/include/desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp#L41.
        matcher_load = LoadGlobalMatcher(arch = self.arch, size = self.size, readonly = False)
        matched_load = matcher_load.match(instructions[offset])

        if matched_load is None:
            return None

        offset += self.collect(matched = matched, new = matched_load)

        # Operation.
        matcher_operation = self.operation.build(load = matched_load)
        matched_operation = matcher_operation.match(instructions = instructions[offset::])

        if matched_operation is None:
            return None

        offset += self.collect(matched = matched, new = matched_operation)

        # Store the value.
        matcher_store = StoreGlobalMatcher(arch = self.arch, size = self.size)
        matched_store = matcher_store.match(instructions[offset])

        if matched_store is None:
            return None

        assert matched_store.operands[1] == matched_load.operands[0]

        offset += self.collect(matched = matched, new = matched_store)

        # The device atomic thread fence block (again).
        matched_thread_fence = DeviceAtomicThreadFenceMatcher.build(arch = self.arch).match(instructions = instructions[offset::])

        if matched_thread_fence is None:
            return None

        offset += self.collect(matched = matched, new = matched_thread_fence)

        # Atomic release.
        matched_atomic_release = AtomicReleaseMatcher.build(arch = self.arch, compiler_id = self.compiler_id).match(inst = instructions[offset])

        if matched_atomic_release is None:
            return None

        self.collect(matched = matched, new = matched_atomic_release)

        return matched
