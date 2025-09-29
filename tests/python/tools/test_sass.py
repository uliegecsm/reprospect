import os
import pathlib

import pytest
import typeguard

from reprospect.tools import sass, binaries

from tests.python.tools.test_binaries import Parameters, PARAMETERS, get_compilation_output

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

FFMA = \
"""\
        /*00c0*/                   FFMA R7, R2, c[0x0][0x160], R7 ;                  /* 0x0000580002077a23 */
                                                                                     /* 0x004fd00000000007 */
"""

ISETP_NE_U32_AND = \
"""\
        /*00c0*/                   ISETP.NE.U32.AND P0, PT, R0.reuse, RZ, PT ;       /* 0x000000ff0000720c */
                                                                                     /* 0x040fe40003f05070 */
"""

class TestSASSDecoder:
    """
    Test :py:class:`reprospect.tools.sass.Decoder`.
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
                hex = '0x0000000004047a24',
                control = sass.ControlCode(stall_count = 5, yield_flag = False, read = 7, write = 7, wait = [True, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
            '        /*0090*/      IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170] ; /* 0x00005c0004047625 */' + '\n' +
            '                                                                /* 0x000fc800078e0005 */' : sass.Instruction(
                offset = 144,
                instruction = 'IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170]',
                hex = '0x00005c0004047625',
                control = sass.ControlCode(stall_count = 4, yield_flag = False, read = 7, write = 7, wait = [False] * 6, reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
        }

        for raw, expt in instructions.items():
            decoder = sass.Decoder(code = raw, skip_until_headerflags = False)

            assert decoder.instructions == [expt], decoder.instructions

        fused = '\n'.join(instructions.keys())
        decoder = sass.Decoder(code = fused, skip_until_headerflags = False)
        assert decoder.instructions == [v for v in instructions.values()], decoder.instructions

    @typeguard.typechecked
    def test_FFMA(self) -> None:
        """
        Check that it can decode `FFMA`.
        """
        decoder = sass.Decoder(code = FFMA, skip_until_headerflags = False)
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
        decoder = sass.Decoder(code = ISETP_NE_U32_AND, skip_until_headerflags = False)
        assert decoder.instructions == [
            sass.Instruction(
                offset = 192, instruction = 'ISETP.NE.U32.AND P0, PT, R0.reuse, RZ, PT',
                hex = '0x000000ff0000720c',
                control = sass.ControlCode(stall_count = 2, yield_flag = True, read = 7, write = 7, wait = [False] * 6, reuse = {'A' : True, 'B' : False, 'C' : False, 'D' : False})
            )
        ], decoder.instructions

    @typeguard.typechecked
    def test_from_source(self) -> None:
        """
        Read `SASS` from a source.
        """
        SOURCE = pathlib.Path(__file__).parent / 'test_sass' / 'saxpy.sass'

        decoder = sass.Decoder(source = SOURCE)

        assert len(decoder.instructions) == 32, len(decoder.instructions)

    @pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
    @typeguard.typechecked
    def test_from_cuobjdump(self, parameters : Parameters) -> None:
        """
        Read `SASS` dumped from `cuobjdump`.
        """
        CUDA_FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'
        output, _ = get_compilation_output(
            source = CUDA_FILE,
            cwd = TMPDIR,
            arch = parameters.arch,
            object = True,
        )

        cuobjdump = binaries.CuObjDump(file = output, arch = parameters.arch, sass = True)

        decoder = sass.Decoder(code = cuobjdump.sass)

        if parameters.arch.compute_capability < 80:
            expt_ninstrs = [16, 16]
        elif parameters.arch.compute_capability < 90:
            expt_ninstrs = [24, 16]
        else:
            expt_ninstrs = [32, 20]

        assert len(decoder.instructions) == expt_ninstrs[0], len(decoder.instructions)

        instructions_not_nop = list(filter(lambda inst: 'NOP' not in inst.instruction, decoder.instructions))

        assert len(instructions_not_nop) == expt_ninstrs[1], len(instructions_not_nop)
