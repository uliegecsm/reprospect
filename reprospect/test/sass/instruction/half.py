import typing

import regex

from reprospect.test.sass.instruction.instruction import PatternMatcher
from reprospect.test.sass.instruction.pattern import PatternBuilder

class Fp16(PatternMatcher):
    """
    Helper for FP16 matchers.

    These instructions typically operate on *packed half-precision pairs* (2 *lanes* per 32-bit register).

    - Each 32-bit register holds two FP16 values (**H0** on bits 0-15 and **H1** on bits 16-31).
    - The instruction can process both lanes in parallel.

    They optionally use *lane selectors*, *e.g.* ``.H0_H0`` or ``.H1_H1``.

    - ``R0.H0_H0`` broadcasts the low half of ``R0`` to both lanes.
    - ``R0.H1_H1`` broadcasts the high half of ``R0`` to both lanes.

    References:

    * :cite:`ho-exploiting-2017`
    """
    REG_LANE : typing.Final[str] = PatternBuilder.REG + r'\.H[01]_H[01]'

    @classmethod
    def reg_lane(cls) -> str:
        return PatternBuilder.group(cls.REG_LANE, group = 'operands')

    @classmethod
    def any_operand(cls) -> str:
        return  PatternBuilder.group(
            PatternBuilder.any(cls.REG_LANE, PatternBuilder.zero_or_more(PatternBuilder.PRE_OPERAND_MOD) + PatternBuilder.REGZ, PatternBuilder.IMMEDIATE),
            group = 'operands',
        )

    @classmethod
    def dst(cls) -> str:
        return PatternBuilder.groups(PatternBuilder.REG, groups = ('dst', 'operands'))

class Fp16FusedMulAddMatcher(Fp16):
    """
    Matcher for 16-bit floating-point fused multiply add (``HFMA2``) instruction, such as::

        HFMA2 R7, R2, R2, -RZ
        HFMA2 R0, R0.H0_H0, R0.H1_H1, 3, 3
        HFMA2 R0, R2, 5, 2, R0.H1_H1

    .. note::

        The FMA computes ``D = A * B + C``. However, it may accept 4 or 5 operands.

        - 4 operands (``HFMA2 D, A, B, C``)

          The addend ``C`` is a register or a single immediate value applied to both lanes.
          Therefore, the following::

            HFMA2 R7, R2, R2, R0

          computes ``R7.H0 = R2.H0 * R2.H0 + R0.H0`` and ``R7.H1 = R2.H1 * R2.H1 + R0.H1``.

        - 5 operands (``HFMA2 E, A, B, C, D``)

          The addend is split per lane with separate immediate values. Therefore, the following::

            HFMA2 R0, R1.H0_H0, R2.H1_H1, 3, 1

          computes ``R0.H0 = R1.H0 * R2.H1 + 3.0`` and ``R0.H1 = R1.H0 * R2.H1 + 1.0``.
    """
    TEMPLATE : typing.Final[str] = f"{PatternBuilder.opcode_mods('HFMA2', modifiers = ('?MMA',))} {Fp16.dst()}, {{op1}}, {{op2}}, {{op3}}(?:, {{op4}})?"

    PATTERN_ANY : typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(
        op1 = Fp16.any_operand(),
        op2 = Fp16.any_operand(),
        op3 = Fp16.any_operand(),
        op4 = Fp16.any_operand(),
    ))

    PATTERN_INDIVIDUAL : typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(
        op1 = Fp16.reg_lane(),
        op2 = Fp16.reg_lane(),
        op3 = PatternBuilder.group(PatternBuilder.any(Fp16.REG_LANE, PatternBuilder.IMMEDIATE), group = 'operands'),
        op4 = PatternBuilder.group(PatternBuilder.any(Fp16.REG_LANE, PatternBuilder.IMMEDIATE), group = 'operands'),
    ))

    PATTERN_PACKED : typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(
        op1 = PatternBuilder.premodregz(),
        op2 = PatternBuilder.premodregz(),
        op3 = PatternBuilder.group(PatternBuilder.any(
            PatternBuilder.zero_or_more(PatternBuilder.PRE_OPERAND_MOD) + PatternBuilder.REGZ,
            PatternBuilder.IMMEDIATE,
        ), group = 'operands'),
        op4 = PatternBuilder.immediate(),
    ))

    def __init__(self, *, packed : bool | None = None) -> None:
        """
        :param packed: If it is packed or not.
        """
        if packed is None:
            pattern = self.PATTERN_ANY
        elif packed is True:
            pattern = self.PATTERN_PACKED
        else:
            pattern = self.PATTERN_INDIVIDUAL
        super().__init__(pattern = pattern)


class Fp16MulMatcher(Fp16):
    """
    Matcher for 16-bit floating-point multiply (``HMUL2``) instruction.

    It may apply on :code:`__half`::

        HMUL2 R0, R2.H0_H0, R3.H0_H0

    or on :code:`__half2`::

        HMUL2 R0, R2, R3
    """
    TEMPLATE : typing.Final[str] = f"{PatternBuilder.opcode_mods('HMUL2')} {Fp16.dst()}, {{op1}}, {{op2}}"

    PATTERN_ANY : typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(
        op1 = Fp16.any_operand(),
        op2 = Fp16.any_operand(),
    ))

    PATTERN_INDIVIDUAL : typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(
        op1 = Fp16.reg_lane(),
        op2 = Fp16.reg_lane(),
    ))

    PATTERN_PACKED : typing.Final[regex.Pattern[str]] = regex.compile(TEMPLATE.format(
        op1 = PatternBuilder.reg(),
        op2 = PatternBuilder.reg(),
    ))

    def __init__(self, *, packed : bool | None = None) -> None:
        """
        :param packed: If it is packed or not.

        References:

        * https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/
        * https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html
        """
        if packed is None:
            pattern = self.PATTERN_ANY
        elif packed is True:
            pattern = self.PATTERN_PACKED
        else:
            pattern = self.PATTERN_INDIVIDUAL
        super().__init__(pattern = pattern)
