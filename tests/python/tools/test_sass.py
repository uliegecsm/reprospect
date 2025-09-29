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
            'IMAD R4, R4, c[0x0][0x0], R3' : sass.Instruction(
                instruction = 'IMAD R4, R4, c[0x0][0x0], R3',
            ),
            'IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170]' : sass.Instruction(
                instruction = 'IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170]',
            ),
        }

        for raw, expt in instructions.items():
            decoder = sass.Decoder(code = raw)

            assert decoder.instructions == [expt]
