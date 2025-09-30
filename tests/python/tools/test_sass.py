import re

import typeguard

from cuda_helpers.tools import sass

FFMA = \
"""\
        /*00c0*/                   FFMA R7, R2, c[0x0][0x160], R7 ;                  /* 0x0000580002077a23 */
                                                                                     /* 0x004fd00000000007 */
"""

IMAD = \
"""\
        /*0040*/      IMAD R4, R4, c[0x0][0x0], R3 ;        /* 0x0000000004047a24 */
                                                            /* 0x001fca00078e0203 */
"""

IMAD_WIDE = \
"""\
        /*0090*/      IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170] ; /* 0x00005c0004047625 */
                                                                /* 0x000fc800078e0005 */
"""

ISETP_NE_U32_AND = \
"""\
        /*00c0*/                   ISETP.NE.U32.AND P0, PT, R0.reuse, RZ, PT ;       /* 0x000000ff0000720c */
                                                                                     /* 0x040fe40003f05070 */
"""

class TestInstruction:
    """
    Test :py:class:`cuda_helpers.tools.sass.Decoder`.
    """
    @typeguard.typechecked
    def test_decode(self):
        """
        Test :py:meth:`cuda_helpers.tools.sass.Decoder.decode`.
        """
        REGEX = r'\/\* ([0-9a-fx]+) \*\/'

        instructions = {
            'FFMA' : FFMA,
            'IMAD' : IMAD,
            'IMAD.WIDE.U32' : IMAD_WIDE,
            'ISETP.NE.U32.AND' : ISETP_NE_U32_AND,
        }

        for op, instruction in instructions.items():
            import logging
            logging.info(instruction)
            decoded = sass.Instruction.decode(code = re.findall(REGEX, instruction)[0])
            assert decoded == {
                'opcode' : op,
            }, decoded

class TestSASSDecoder:
    """
    Test :py:class:`cuda_helpers.tools.sass.Decoder`.
    """
    @typeguard.typechecked
    def test_IMAD(self) -> None:
        """
        Check that it can decode `IMAD`.
        """
        instructions = {
            IMAD : sass.Instruction(
                offset = 64,
                instruction = 'IMAD R4, R4, c[0x0][0x0], R3',
                hex = '0x0000000004047a24',
                control = sass.ControlCode(stall_count = 5, yield_flag = False, read = 7, write = 7, wait = [True, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
            IMAD_WIDE : sass.Instruction(
                offset = 144,
                instruction = 'IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170]',
                hex = '0x00005c0004047625',
                control = sass.ControlCode(stall_count = 4, yield_flag = False, read = 7, write = 7, wait = [False] * 6, reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
        }

        for raw, expt in instructions.items():
            decoder = sass.Decoder(code = raw)

            assert decoder.instructions == [expt], decoder.instructions

        fused = ''.join(instructions.keys())
        decoder = sass.Decoder(code = fused)
        assert decoder.instructions == [v for v in instructions.values()], decoder.instructions

    @typeguard.typechecked
    def test_FFMA(self) -> None:
        """
        Check that it can decode `FFMA`.
        """
        decoder = sass.Decoder(code = FFMA)
        assert decoder.instructions == [
            sass.Instruction(
                offset = 192, instruction = 'FFMA R7, R2, c[0x0][0x160], R7',
                hex = '0x0000580002077a23',
                control = sass.ControlCode(stall_count = 8, yield_flag = False, read = 7, write = 7, wait = [False, False, True, False, False, False], reuse = {'A': False, 'B': False, 'C': False, 'D': False}))
        ], decoder.instructions

    @typeguard.typechecked
    def test_ISETP(self) -> None:
        """
        Check that it can decode `ISETP`.
        """
        decoder = sass.Decoder(code = ISETP_NE_U32_AND)
        assert decoder.instructions == [
            sass.Instruction(
                offset = 192, instruction = 'ISETP.NE.U32.AND P0, PT, R0.reuse, RZ, PT',
                hex = '0x000000ff0000720c',
                control = sass.ControlCode(stall_count = 2, yield_flag = True, read = 7, write = 7, wait = [False] * 6, reuse = {'A' : True, 'B' : False, 'C' : False, 'D' : False})
            )
        ], decoder.instructions
