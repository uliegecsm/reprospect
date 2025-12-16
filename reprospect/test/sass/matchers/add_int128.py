import sys
import typing

from reprospect.test.sass.composite_impl import SequenceMatcher
from reprospect.test.sass.instruction import InstructionMatch, OpcodeModsWithOperandsMatcher, PatternBuilder, RegisterMatcher
from reprospect.tools.sass.decode import Instruction

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class AddInt128(SequenceMatcher):
    """
    Match SASS instructions originating from the addition of 2 :code:`__int128`.

    It may use one of several instruction patterns:

    * :py:meth:`pattern_3IADD`
    * :py:meth:`pattern_4IADD3`

    .. note::

        For a simple "load, add, store" sequence, the PTX may look as follows::

            ld.global.nc.v2.u64 {%rd7, %rd8}, [%rd6];
            add.s64 %rd9, %rd3, %rd5;
            ld.global.v2.u64 {%rd10, %rd11}, [%rd9];
            add.cc.s64 %rd12, %rd10, %rd7;
            addc.cc.s64 %rd13, %rd11, %rd8;
            st.global.v2.u64 [%rd9], {%rd12, %rd13};

        According to https://docs.nvidia.com/cuda/parallel-thread-execution/#extended-precision-arithmetic-instructions-add-cc, the carry-out
        value is written in the condition code register, usually ``P0``.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('_index',)

    def __init__(self) -> None:
        self._index: int = 0

    def pattern_3IADD(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None: # pylint: disable=invalid-name
        """
        Typically::

            IADD.64 RZ, P0, R4, UR12
            IADD.64 R4, R4, UR12
            IADD.64.X R6, R6, UR14, P0
        """
        matched : list[InstructionMatch] = []

        matcher_iadd_step_0 = OpcodeModsWithOperandsMatcher(
            opcode = 'IADD', modifiers = ('64',),
            operands = (
                'RZ',
                PatternBuilder.PRED,
                PatternBuilder.anygpreg(reuse = None, group = 'start'),
                PatternBuilder.anygpreg(reuse = None),
            ),
        )
        if (matched_iadd_step_0 := matcher_iadd_step_0.match(instructions[0])) is None:
            return None

        matched.append(matched_iadd_step_0)

        iadd_step_0_reg_0 = RegisterMatcher(special = False).match(matched_iadd_step_0.operands[-2])
        assert iadd_step_0_reg_0 is not None and iadd_step_0_reg_0.index is not None
        iadd_step_0_reg_1 = RegisterMatcher(special = False).match(matched_iadd_step_0.operands[-1])
        assert iadd_step_0_reg_1 is not None and iadd_step_0_reg_1.index is not None

        matcher_iadd_step_1 = OpcodeModsWithOperandsMatcher(
            opcode = 'IADD', modifiers = ('64',),
            operands = (
                f'{iadd_step_0_reg_0.rtype}{iadd_step_0_reg_0.index}',
                f'{iadd_step_0_reg_0.rtype}{iadd_step_0_reg_0.index}',
                f'{iadd_step_0_reg_1.rtype}{iadd_step_0_reg_1.index}',
            ),
        )
        if (matched_iadd_step_1 := matcher_iadd_step_1.match(instructions[1])) is None:
            return None

        matched.append(matched_iadd_step_1)

        iadd_step_2_reg_0 = f'{iadd_step_0_reg_0.rtype}{iadd_step_0_reg_0.index + 2}'
        iadd_step_2_reg_1 = f'{iadd_step_0_reg_1.rtype}{iadd_step_0_reg_1.index + 2}'

        matcher_iadd_step_2 = OpcodeModsWithOperandsMatcher(
            opcode = 'IADD', modifiers = ('64', 'X'),
            operands = (
                iadd_step_2_reg_0,
                iadd_step_2_reg_0,
                iadd_step_2_reg_1,
                matched_iadd_step_0.operands[1],
            ),
        )
        if (matched_iadd_step_2 := matcher_iadd_step_2.match(instructions[2])) is None:
            return None

        matched.append(matched_iadd_step_2)

        self._index = 3

        return matched

    def pattern_4IADD3(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None: # pylint: disable=invalid-name
        """
        Typically::

            IADD3 R8, P0, R8, c[0x0][0x180], RZ
            IADD3.X R9, P0, R9, c[0x0][0x184], RZ, P0, !PT
            IADD3.X R10, P0, R10, c[0x0][0x188], RZ, P0, !PT
            IADD3.X R11, R11, c[0x0][0x18c], RZ, P0, !PT

        or::

            IADD3 R4, P0, PT, R4, R12, RZ
            IADD3.X R5, P0, PT, R5, R13, RZ, P0, !PT
            IADD3.X R6, P0, PT, R6, R14, RZ, P0, !PT
            IADD3.X R7, PT, PT, R7, R15, RZ, P0, !PT

        That is, there may be one additional argument.
        """
        matched : list[InstructionMatch] = []

        matcher_iadd3_step_0 = OpcodeModsWithOperandsMatcher(
            opcode = 'IADD3',
            operands = (
                PatternBuilder.group(PatternBuilder.REG, group = 'start'),
                PatternBuilder.PRED,
                PatternBuilder.zero_or_one(PatternBuilder.PREDT),
                PatternBuilder.REG,
                PatternBuilder.any(PatternBuilder.REG, PatternBuilder.CONSTANT),
                'RZ',
            ),
        )
        if (matched_iadd3_step_0 := matcher_iadd3_step_0.match(instructions[0])) is None:
            return None

        if matched_iadd3_step_0.operands.count(matched_iadd3_step_0.operands[0]) != 2:
            return None

        matched.append(matched_iadd3_step_0)

        iadd3_step_0_reg = RegisterMatcher(special = False).match(matched_iadd3_step_0.operands[0])
        assert iadd3_step_0_reg is not None and iadd3_step_0_reg.index is not None
        iadd3_step_1_reg = f'{iadd3_step_0_reg.rtype}{iadd3_step_0_reg.index + 1}'

        matcher_iadd3_step_1 = OpcodeModsWithOperandsMatcher(
            opcode = 'IADD3', modifiers = ('X',),
            operands = (
                iadd3_step_1_reg,
                PatternBuilder.PRED,
                PatternBuilder.zero_or_one(PatternBuilder.PREDT),
                iadd3_step_1_reg,
                PatternBuilder.any(PatternBuilder.REG, PatternBuilder.CONSTANT),
                'RZ',
                PatternBuilder.PRED,
                '!PT',
            ),
        )
        if (matched_iadd3_step_1 := matcher_iadd3_step_1.match(instructions[1])) is None:
            return None

        matched.append(matched_iadd3_step_1)

        iadd3_step_2_reg = f'{iadd3_step_0_reg.rtype}{iadd3_step_0_reg.index + 2}'

        matcher_iadd3_step_2 = OpcodeModsWithOperandsMatcher(
            opcode = 'IADD3', modifiers = ('X',),
            operands = (
                iadd3_step_2_reg,
                PatternBuilder.PRED,
                PatternBuilder.zero_or_one(PatternBuilder.PREDT),
                iadd3_step_2_reg,
                PatternBuilder.any(PatternBuilder.REG, PatternBuilder.CONSTANT),
                'RZ',
                PatternBuilder.PRED,
                '!PT',
            ),
        )
        if (matched_iadd3_step_2 := matcher_iadd3_step_2.match(instructions[2])) is None:
            return None

        matched.append(matched_iadd3_step_2)

        iadd3_step_3_reg = f'{iadd3_step_0_reg.rtype}{iadd3_step_0_reg.index + 3}'

        matcher_iadd3_step_3 = OpcodeModsWithOperandsMatcher(
            opcode = 'IADD3', modifiers = ('X',),
            operands = (
                iadd3_step_3_reg,
                PatternBuilder.zero_or_one(PatternBuilder.PREDT),
                PatternBuilder.zero_or_one(PatternBuilder.PREDT),
                iadd3_step_3_reg,
                PatternBuilder.any(PatternBuilder.REG, PatternBuilder.CONSTANT),
                'RZ',
                PatternBuilder.PRED,
                '!PT',
            ),
        )
        if (matched_iadd3_step_3 := matcher_iadd3_step_3.match(instructions[3])) is None:
            return None

        matched.append(matched_iadd3_step_3)

        self._index = 4

        return matched

    @override
    def match(self, instructions : typing.Sequence[Instruction | str]) -> list[InstructionMatch] | None:
        instruction = instructions[0].instruction if isinstance(instructions[0], Instruction) else instructions[0]
        if instruction.startswith('IADD.64'):
            return self.pattern_3IADD(instructions = instructions)
        if instruction.startswith('IADD3'):
            return self.pattern_4IADD3(instructions = instructions)
        return None

    @override
    @property
    def next_index(self) -> int:
        return self._index
