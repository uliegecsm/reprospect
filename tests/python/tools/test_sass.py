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
            '/*0040*/                   IMAD R4, R4, c[0x0][0x0], R3' : sass.Instruction(
                offset = 64,
                instruction = 'IMAD R4, R4, c[0x0][0x0], R3',
            ),
            '/*0090*/                   IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170]' : sass.Instruction(
                offset = 144,
                instruction = 'IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170]',
            ),
        }

        for raw, expt in instructions.items():
            decoder = sass.Decoder(code = raw)

            assert decoder.instructions == [expt]
