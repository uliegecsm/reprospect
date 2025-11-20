import logging
import os
import pathlib
import re

import pytest

from reprospect.tools import sass, binaries
from reprospect.utils import cmake

from tests.python.compilation import get_compilation_output
from tests.python.cublas      import CuBLAS
from tests.python.parameters  import Parameters, PARAMETERS

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

class TestSASSDecoder:
    """
    Test :py:class:`reprospect.tools.sass.Decoder`.
    """
    def test_matchers(self) -> None:
        """
        Simple tests for the matchers.
        """
        assert re.match(sass.Decoder.HEX, '0x00000a0000017a02') is not None

        # Complete SASS line (offset, instruction and hex), without noise.
        matched = re.match(sass.Decoder.MATCHER, '/*0090*/                   ISETP.GE.U32.AND P0, PT, R0, UR9, PT ;           /* 0x0000000900007c0c */')
        assert matched is not None
        assert matched.group(1) == '0090'
        assert matched.group(2) == 'ISETP.GE.U32.AND P0, PT, R0, UR9, PT '
        assert matched.group(3) == '0x0000000900007c0c'

        matched = re.match(sass.Decoder.MATCHER, '/*01d0*/              @!P0 BRA 0x100 ;                                      /* 0xfffffffc00c88947 */')
        assert matched is not None
        assert matched.group(1) == '01d0'
        assert matched.group(2) == '@!P0 BRA 0x100 '
        assert matched.group(3) == '0xfffffffc00c88947'

        # With noise.
        matched = re.match(sass.Decoder.MATCHER, '/*00e0*/                   LDC.64 R10, c[0x0][0x3a0]                     &wr=0x2               ?trans8;           /* 0x0000e800ff0a7b82 */')
        assert matched is not None
        assert matched.group(1) == '00e0'
        assert matched.group(2) == 'LDC.64 R10, c[0x0][0x3a0]'
        assert matched.group(3) == '0x0000e800ff0a7b82'

        matched = re.match(sass.Decoder.MATCHER, '/*00b0*/                   ATOM.E.ADD.F64.RN.STRONG.SM P0, RZ, desc[UR4][R2.64], R4  &wr_early=0x1 ;           /* 0x8000000402ff79a2 */')
        assert matched is not None
        assert matched.group(1) == '00b0'
        assert matched.group(2) == 'ATOM.E.ADD.F64.RN.STRONG.SM P0, RZ, desc[UR4][R2.64], R4'
        assert matched.group(3) == '0x8000000402ff79a2'

    def test_IMAD(self) -> None:
        """
        Check that it can decode `IMAD`.
        """
        instructions = {
            IMAD : sass.Instruction(
                offset = int('0040', base = 16),
                instruction = 'IMAD R4, R4, c[0x0][0x0], R3',
                hex = '0x0000000004047a24',
                control = sass.ControlCode(stall_count = 5, yield_flag = False, read = 7, write = 7, wait = [True, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
            IMAD_WIDE_U32 : sass.Instruction(
                offset = int('0090', base = 16),
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
        assert decoder.instructions == list(instructions.values()), decoder.instructions

    def test_FFMA(self) -> None:
        """
        Check that it can decode `FFMA`.
        """
        decoder = sass.Decoder(code = FFMA, skip_until_headerflags = False)
        assert decoder.instructions == [
            sass.Instruction(
                offset = int('00c0', base = 16), instruction = 'FFMA R7, R2, c[0x0][0x160], R7',
                hex = '0x0000580002077a23',
                control = sass.ControlCode(stall_count = 8, yield_flag = False, read = 7, write = 7, wait = [False, False, True, False, False, False], reuse = {'A': False, 'B': False, 'C': False, 'D': False}))
        ], decoder.instructions

    def test_ISETP(self) -> None:
        """
        Check that it can decode `ISETP`.
        """
        decoder = sass.Decoder(code = ISETP_NE_U32_AND, skip_until_headerflags = False)
        assert decoder.instructions == [
            sass.Instruction(
                offset = int('00c0', base = 16), instruction = 'ISETP.NE.U32.AND P0, PT, R0.reuse, RZ, PT',
                hex = '0x000000ff0000720c',
                control = sass.ControlCode(stall_count = 2, yield_flag = True, read = 7, write = 7, wait = [False] * 6, reuse = {'A' : True, 'B' : False, 'C' : False, 'D' : False})
            )
        ], decoder.instructions

    def test_from_source(self) -> None:
        """
        Read SASS from a source.
        """
        SOURCE = pathlib.Path(__file__).parent / 'assets' / 'saxpy.sass'

        decoder = sass.Decoder(source = SOURCE)

        assert len(decoder.instructions) == 32, len(decoder.instructions)

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_from_cuobjdump(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Read SASS dumped from ``cuobjdump``.
        """
        CUDA_FILE = pathlib.Path(__file__).parent / 'assets' / 'saxpy.cu'
        output, _ = get_compilation_output(
            source = CUDA_FILE,
            cwd = workdir,
            arch = parameters.arch,
            object_file = True,
            cmake_file_api = cmake_file_api,
        )

        cuobjdump = binaries.CuObjDump(file = output, arch = parameters.arch, sass = True)

        assert len(cuobjdump.functions) == 1

        decoder = sass.Decoder(code = next(iter(cuobjdump.functions.values())).code)

        match parameters.arch.compute_capability.as_int:
            case 70:
                expt_ninstrs = [16, 16]
            case 75:
                expt_ninstrs = [16, 15]
            case 80 | 86 | 89:
                expt_ninstrs = [24, 16]
            case 90 | 100 | 120:
                expt_ninstrs = [32, 20]
            case _:
                raise ValueError(f'unsupported {parameters.arch.compute_capability}')

        assert len(decoder.instructions) == expt_ninstrs[0], len(decoder.instructions)

        instructions_not_nop = list(filter(lambda inst: 'NOP' not in inst.instruction, decoder.instructions))

        assert len(instructions_not_nop) == expt_ninstrs[1], len(instructions_not_nop)

    def test_string_representation(self) -> None:
        """
        Test the string representation :py:class:`reprospect.tools.sass.Decoder`.
        """
        decoder = sass.Decoder()

        decoder.instructions = [
            sass.Instruction(
                offset = int('00f0', base = 16),
                instruction = 'LDC R1, c[0x0][0x37c]',
                hex = '0x0000df00ff017b82',
                control = sass.ControlCode(stall_count = 1, yield_flag = True, read = 7, write = 0, wait = [True, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
            sass.Instruction(
                offset = int('0190', base = 16),
                instruction = 'S2R R0, SR_TID.X',
                hex = '0x0000000000007919',
                control = sass.ControlCode(stall_count = 7, yield_flag = False, read = 7, write = 1, wait = [False, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
            sass.Instruction(
                offset = int('01a0', base = 16),
                instruction = '@P0 EXIT',
                hex = '0x000000000000094d',
                control = sass.ControlCode(stall_count = 5, yield_flag = True, read = 2, write = 7, wait = [True, False, False, False, False, False], reuse = {'A' : False, 'B' : False, 'C' : False, 'D' : False}),
            ),
        ]

        assert str(decoder) == """\
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┓
┃ offset ┃ instruction           ┃ stall ┃ yield ┃ b0    ┃ b1 ┃ b2 ┃ b3 ┃ b4 ┃ b5 ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━┩
│ 00f0   │ LDC R1, c[0x0][0x37c] │ 1     │ True  │ Wa/Wr │    │    │    │    │    │
│ 0190   │ S2R R0, SR_TID.X      │ 7     │ False │       │ Wr │    │    │    │    │
│ 01a0   │ @P0 EXIT              │ 5     │ True  │ Wa    │    │ Re │    │    │    │
└────────┴───────────────────────┴───────┴───────┴───────┴────┴────┴────┴────┴────┘
"""

    def test_to_html(self) -> None:
        """
        Test :py:meth:`reprospect.tools.sass.Decoder.to_html`.
        """
        ARTIFACT_DIR = pathlib.Path(os.environ['ARTIFACT_DIR'])
        ARTIFACT_DIR.mkdir(parents = True, exist_ok = True)

        SOURCE = pathlib.Path(__file__).parent / 'assets' / 'saxpy.sass'

        decoder = sass.Decoder(source = SOURCE)

        file = ARTIFACT_DIR / 'saxpy.html'

        logging.info(f'Writing SASS decoded table to {file}.')

        with file.open('w+') as fout:
            fout.write(decoder.to_html())

    def test_from_128_to_130(self) -> None:
        """
        This test is related to an observation that, for the same
        architecture (`120`), `nvcc 12.8.1` and `nvcc 13.0.0` write the same sequence of instructions,
        but the barriers (scoreboard dependencies) are different.
        """
        cfd = pathlib.Path(__file__).parent

        d1281 = sass.Decoder(source = cfd / 'assets' / '12.8.1.sass')
        d1300 = sass.Decoder(source = cfd / 'assets' / '13.0.0.sass')

        assert len(d1281.instructions) == len(d1300.instructions)

        current = True

        for d1281i, d1300i in zip(d1281.instructions, d1300.instructions):
            assert d1281i.instruction == d1300i.instruction
            assert d1281i.control.stall_count == d1300i.control.stall_count
            assert d1281i.control.yield_flag == d1300i.control.yield_flag
            assert d1281i.control.reuse == d1300i.control.reuse

            current = current and (d1281i.control == d1300i.control)

        assert current is False

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_cuBLAS(self, parameters : Parameters, workdir : pathlib.Path, cmake_file_api : cmake.FileAPI) -> None:
        cublas = CuBLAS(cmake_file_api = cmake_file_api)

        try:
            [cubin] = cublas.extract(arch = parameters.arch, cwd = workdir, randomly = True)
        except IndexError:
            pytest.skip(f'The library {cublas.libcublas} does not contain any CUDA binary file for {parameters.arch}.')

        assert cubin.is_file()

        cuobjdump = binaries.CuObjDump(file = cubin, arch = parameters.arch, sass = True)

        for name, function in cuobjdump.functions.items():
            decoder = sass.Decoder(code = function.code)
            assert len(decoder.instructions) > 1
            logging.info(f'Function {name} in {cublas.libcublas} has {len(decoder.instructions)} SASS instructions.')
