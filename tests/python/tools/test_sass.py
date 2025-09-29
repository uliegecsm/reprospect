import typeguard

from cuda_helpers.tools import sass

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
            '        /*0040*/      IMAD R4, R4, c[0x0][0x0], R3 ;        /* 0x0000000004047a24 */' + '\n' +
            '                                                            /* 0x001fca00078e0203 */' : sass.Instruction(
                offset = 64,
                instruction = 'IMAD R4, R4, c[0x0][0x0], R3',
                hex = ['0x0000000004047a24', '0x001fca00078e0203'],
            ),
            '        /*0090*/      IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170] ; /* 0x00005c0004047625 */' + '\n' +
            '                                                                /* 0x000fc800078e0005 */' : sass.Instruction(
                offset = 144,
                instruction = 'IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170]',
                hex = ['0x00005c0004047625', '0x000fc800078e0005'],
            ),
        }

        for raw, expt in instructions.items():
            decoder = sass.Decoder(code = raw)

            assert decoder.instructions == [expt]
