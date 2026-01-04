import re
import typing

import pytest

from reprospect.test.sass.instruction import OpcodeModsWithOperandsMatcher
from reprospect.test.sass.instruction.immediate import Immediate
from reprospect.test.sass.instruction.operand import Operand
from reprospect.test.sass.instruction.register import Register


class TestOperand:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.operand.Operand`.
    """

    OPERANDS: typing.Final[tuple[str, ...]] = (
        'SR_TID.X',
        'R126',
        'UR24',
        'R125.reuse',
        '-R128',
        '~URZ',
        '|R165|',
        '!PT',
        '[R135]',
        '[R155+0x1200]',
        '[R192.X8+0x2000]',
        'desc[UR22][R10.64]',
        'desc[UR22][R16.64+0x80]',
        'c[0x0][0x358]',
        '100',
        '0xffffffe0',
        '-0xfe',
        '+INF',
    )
    """
    Zoo of real operands.
    """

    def test_mod(self) -> None:
        assert re.match(Operand.mod(Register.REG, math=None), 'R42') is not None

        assert re.match(Operand.mod(Register.REG, math=None), '|R165|') is not None

        assert re.match(Operand.mod(Register.UREGZ, math=True), '~URZ') is not None

        assert re.match(Operand.mod(Register.REG, math=False), '-R42') is None

        assert OpcodeModsWithOperandsMatcher(
            opcode='FSETP',
            modifiers=('GEU', 'AND'),
            operands=(Register.PREDT, Register.PREDT, Operand.mod(Register.REG, math=None), Immediate.FLOATING_OR_LIMIT, Register.PREDT),
        ).match('FSETP.GEU.AND P1, PT, |R151|, +INF, PT') is not None

    @pytest.mark.parametrize('opnd', OPERANDS)
    def test_any(self, opnd: str) -> None:
        assert re.match(Operand.OPERAND, opnd) is not None
