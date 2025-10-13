import os
import pathlib

import pytest
import typeguard

from reprospect.utils import cmake
from reprospect.tools import sass, binaries

from tests.python.tools.test_binaries import Parameters, PARAMETERS, get_compilation_output

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

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

IMAD_WIDE_U32 = \
"""\
        /*0090*/      IMAD.WIDE.U32 R4, R4, R5, c[0x0][0x170] ; /* 0x00005c0004047625 */
                                                                /* 0x000fc800078e0005 */
"""

ISETP_NE_U32_AND = \
"""\
        /*00c0*/                   ISETP.NE.U32.AND P0, PT, R0.reuse, RZ, PT ;       /* 0x000000ff0000720c */
                                                                                     /* 0x040fe40003f05070 */
"""

@pytest.fixture(scope = 'session')
@typeguard.typechecked
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        build_path = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
        inspect = {'cache' : 2, 'toolchains' : 1},
    )

class TestSASSDecoder:
    """
    Test :py:class:`reprospect.tools.sass.Decoder`.
    """
    def test_IMAD(self):
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
            IMAD_WIDE_U32 : sass.Instruction(
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

    def test_FFMA(self):
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

    def test_ISETP(self):
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

    def test_from_source(self):
        """
        Read `SASS` from a source.
        """
        SOURCE = pathlib.Path(__file__).parent / 'test_sass' / 'saxpy.sass'

        decoder = sass.Decoder(source = SOURCE)

        assert len(decoder.instructions) == 32, len(decoder.instructions)

    @pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
    @typeguard.typechecked
    def test_from_cuobjdump(self, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Read `SASS` dumped from `cuobjdump`.
        """
        CUDA_FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'
        output, _ = get_compilation_output(
            source = CUDA_FILE,
            cwd = TMPDIR,
            arch = parameters.arch,
            object = True,
            cmake_file_api = cmake_file_api,
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

    def test_string_representation(self):
        """
        Test :py:meth:`reprospect.tools.sass.Decoder.__str__`.
        """
        decoder = sass.Decoder()

        decoder.instructions = [
            sass.Instruction(
                offset = 0,
                instruction = 'LDC R1, c[0x0][0x37c]',
                hex = '0x0000df00ff017b82',
                control = sass.ControlCode(stall_count = 1, yield_flag = True, read = 7, write = 0, wait = [False, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
            sass.Instruction(
                offset = 16,
                instruction = 'S2R R0, SR_TID.X',
                hex = '0x0000000000007919',
                control = sass.ControlCode(stall_count = 7, yield_flag = False, read = 7, write = 1, wait = [False, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
            sass.Instruction(
                offset = 112,
                instruction = '@P0 EXIT',
                hex = '0x000000000000094d',
                control = sass.ControlCode(stall_count = 5, yield_flag = True, read = 2, write = 7, wait = [True, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
        ]

        assert str(decoder) == """\
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┓
┃ offset ┃ instruction           ┃ stall ┃ yield ┃ b0 ┃ b1 ┃ b2 ┃ b3 ┃ b4 ┃ b5 ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━┩
│ 0      │ LDC R1, c[0x0][0x37c] │ 1     │ True  │ Wr │    │    │    │    │    │
│ 16     │ S2R R0, SR_TID.X      │ 7     │ False │    │ Wr │    │    │    │    │
│ 112    │ @P0 EXIT              │ 5     │ True  │ Wa │    │ Re │    │    │    │
└────────┴───────────────────────┴───────┴───────┴────┴────┴────┴────┴────┴────┘
"""

    def test_to_html(self):
        """
        Test :py:meth:`reprospect.tools.sass.Decoder.to_html`.
        """
        ARTIFACT_DIR = pathlib.Path(os.environ['ARTIFACT_DIR'])
        ARTIFACT_DIR.mkdir(parents = True, exist_ok = True)

        SOURCE = pathlib.Path(__file__).parent / 'test_sass' / 'saxpy.sass'

        decoder = sass.Decoder(source = SOURCE)

        with open(ARTIFACT_DIR / 'saxpy.html', 'w+') as fout:
            fout.write(decoder.to_html())

    def test_from_128_to_130(self):
        """
        This test is related to an observation that, for the same
        architecture (`120`), `nvcc 12.8.1` and `nvcc 13.0.0` write the same sequence of instructions,
        but the barriers (scoreboard dependencies) are different.
        """
        cfd = pathlib.Path(__file__).parent

        d1281 = sass.Decoder(source = cfd / 'test_sass' / '12.8.1.sass')
        d1300 = sass.Decoder(source = cfd / 'test_sass' / '13.0.0.sass')

        assert len(d1281.instructions) == len(d1300.instructions)

        current = True

        for d1281i, d1300i in zip(d1281.instructions, d1300.instructions):
            assert d1281i.instruction == d1300i.instruction
            assert d1281i.control.stall_count == d1300i.control.stall_count
            assert d1281i.control.yield_flag == d1300i.control.yield_flag
            assert d1281i.control.reuse == d1300i.control.reuse

            current = current and (d1281i.control == d1300i.control)

        assert current is False
