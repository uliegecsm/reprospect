import os
import pathlib
import typing

import pytest

from reprospect.tools.sass.controlflow import BasicBlock, ControlFlow, Graph
from reprospect.tools.sass.decode      import ControlCode, Decoder, Instruction
from reprospect.utils                  import cmake

from tests.python.parameters  import Parameters, PARAMETERS

from tests.python.test.sass.test_instruction import get_decoder

class TestGraph:
    """
    Tests for :py:class:`reprospect.tools.sass.controlflow.Graph`.
    """
    CONTROLCODE : typing.Final[ControlCode] = ControlCode.decode(code = '0')

    INSTRUCTIONS : typing.Final[tuple[Instruction, ...]] = (
        Instruction(offset = 0, instruction = 'DADD R4, R4, c[0x0][0x180]', hex = '0x0', control = CONTROLCODE),
        Instruction(offset = 0, instruction = 'FADD R4, R4, c[0x0][0x180]', hex = '0x0', control = CONTROLCODE),
        Instruction(offset = 0, instruction = 'DMUL R6, R6, c[0x0][0x188]', hex = '0x1', control = CONTROLCODE),
        Instruction(offset = 0, instruction = 'NOP',                        hex = '0x2', control = CONTROLCODE),
    )

    @pytest.fixture(scope = 'function')
    def cfg(self) -> Graph:
        return Graph()

    def test_add_block(self, cfg : Graph) -> None:
        """
        Add a block.
        """
        cfg.add_block(block = BasicBlock(self.INSTRUCTIONS))

        assert len(cfg.blocks) == 1

    def test_add_blocks(self, cfg : Graph) -> None:
        """
        Add blocks.
        """
        block_0 = BasicBlock(self.INSTRUCTIONS[0:2])
        block_1 = BasicBlock(self.INSTRUCTIONS[2:4])

        cfg.add_blocks(blocks = (block_0, block_1))

        assert len(cfg.blocks) == 2

    def test_add_blocks_with_edges(self, cfg : Graph) -> None:
        """
        Add blocks with edges.
        """
        block_0 = BasicBlock(self.INSTRUCTIONS[0:2])
        block_1 = BasicBlock(self.INSTRUCTIONS[2:4])

        cfg.add_blocks(blocks = (block_0, block_1))

        cfg.add_edge(src = block_0, dst = block_1)

        assert cfg.edges == {block_0 : {block_1}}

    def test_to_mermaid(self, cfg : Graph) -> None:
        """
        Convert to mermaid.
        """
        block_0 = BasicBlock(self.INSTRUCTIONS[0:2])
        block_1 = BasicBlock(self.INSTRUCTIONS[2:4])

        cfg.add_blocks(blocks = (block_0, block_1))

        cfg.add_edge(src = block_0, dst = block_1)
        cfg.add_edge(src = block_1, dst = block_0)

        assert cfg.to_mermaid() == f"""\
---
title : CFG
---
flowchart TD
\tblock_{self.INSTRUCTIONS[0].offset}["{self.INSTRUCTIONS[0].instruction}<br/>{self.INSTRUCTIONS[1].instruction}"]:::myblock
\tblock_{self.INSTRUCTIONS[2].offset}["{self.INSTRUCTIONS[2].instruction}<br/>{self.INSTRUCTIONS[3].instruction}"]:::myblock
\tblock_{self.INSTRUCTIONS[0].offset} --> block_{self.INSTRUCTIONS[2].offset}
\tblock_{self.INSTRUCTIONS[2].offset} --> block_{self.INSTRUCTIONS[0].offset}
\tclassDef myblock text-align:left\
"""

class TestControlFlow:
    """
    Tests for :py:class:`reprospect.tools.sass.controlflow.ControlFlow`.
    """
    class TestIfs:
        CU_IFS         : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'ifs.cu'
        SASS_IFS_SM120 : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'ifs.sm_120.sass'
        INSTRUCTIONS   : typing.Final[list[Instruction]] = Decoder(source = SASS_IFS_SM120).instructions

        def test_find_entry_points(self) -> None:
            assert ControlFlow.find_entry_points(instructions = self.INSTRUCTIONS) == {
                int('0000', base = 16),
                int('0x130', base = 16),
                int('0090', base = 16),
                int('0100', base = 16),
                int('0130', base = 16),
                int('0150', base = 16),
                int('0x1a0', base = 16),
                int('01a0', base = 16),
                int('01b0', base = 16),
            }

        def test_create_blocks(self) -> None:
            entry_points = ControlFlow.find_entry_points(instructions = self.INSTRUCTIONS)
            blocks = ControlFlow.create_blocks(instructions = self.INSTRUCTIONS, entry_points = entry_points)

            assert len(blocks) == 6

            assert len(blocks[0].instructions) == 9
            assert blocks[0].instructions == tuple(self.INSTRUCTIONS[0:9])

            assert len(blocks[1].instructions) == 7
            assert blocks[1].instructions == tuple(self.INSTRUCTIONS[9:16])

            assert len(blocks[2].instructions) == 3
            assert blocks[2].instructions == tuple(self.INSTRUCTIONS[16:19])

            assert len(blocks[3].instructions) == 2
            assert blocks[3].instructions == tuple(self.INSTRUCTIONS[19:21])

            assert len(blocks[4].instructions) == 5
            assert blocks[4].instructions == tuple(self.INSTRUCTIONS[21:26])

            assert len(blocks[5].instructions) == 1
            assert blocks[5].instructions == tuple(self.INSTRUCTIONS[26:27])

        def test_analyse(self) -> None:
            cfg = ControlFlow.analyze(instructions = self.INSTRUCTIONS)

            assert cfg.to_mermaid() == f"""\
---
title : CFG
---
flowchart TD
\tblock_{self.INSTRUCTIONS[ 0].offset}["{'<br/>'.join(instr.instruction for instr in self.INSTRUCTIONS[0:9])}"]:::myblock
\tblock_{self.INSTRUCTIONS[ 9].offset}["{'<br/>'.join(instr.instruction for instr in self.INSTRUCTIONS[9:16])}"]:::myblock
\tblock_{self.INSTRUCTIONS[16].offset}["{'<br/>'.join(instr.instruction for instr in self.INSTRUCTIONS[16:19])}"]:::myblock
\tblock_{self.INSTRUCTIONS[19].offset}["{'<br/>'.join(instr.instruction for instr in self.INSTRUCTIONS[19:21])}"]:::myblock
\tblock_{self.INSTRUCTIONS[21].offset}["{'<br/>'.join(instr.instruction for instr in self.INSTRUCTIONS[21:26])}"]:::myblock
\tblock_{self.INSTRUCTIONS[26].offset}["{'<br/>'.join(instr.instruction for instr in self.INSTRUCTIONS[26:27])}"]:::myblock
\tblock_{self.INSTRUCTIONS[ 0].offset} --> block_{self.INSTRUCTIONS[19].offset}
\tblock_{self.INSTRUCTIONS[ 0].offset} --> block_{self.INSTRUCTIONS[ 9].offset}
\tblock_{self.INSTRUCTIONS[ 9].offset} --> block_{self.INSTRUCTIONS[16].offset}
\tblock_{self.INSTRUCTIONS[19].offset} --> block_{self.INSTRUCTIONS[21].offset}
\tblock_{self.INSTRUCTIONS[26].offset} --> block_{self.INSTRUCTIONS[26].offset}
\tclassDef myblock text-align:left\
"""

        @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
        def test(self, workdir : pathlib.Path, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CU_IFS` and build the CFG.
            """
            decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = self.CU_IFS, cmake_file_api = cmake_file_api)
            cfg = ControlFlow.analyze(instructions = decoder.instructions)
            assert len(cfg.blocks) == 6

    class TestAtomicAddInt64:
        SASS_ATOM_SM120 : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'atomic_add.int64.sm_120.sass'

        def test(self) -> None: # pylint: disable=too-many-statements
            """
            Parse :py:attr:`SASS_ATOM_SM120`.
            """
            decoder = Decoder(source = self.SASS_ATOM_SM120)
            cfg = ControlFlow.analyze(instructions = decoder.instructions)

            ARTIFACT_DIR = pathlib.Path(os.environ['ARTIFACT_DIR'])
            ARTIFACT_DIR.mkdir(parents = True, exist_ok = True)

            (ARTIFACT_DIR / 'test_atomic_add_int64.mmd').write_text(cfg.to_mermaid())

            # Block 0.
            assert len(cfg.blocks[0].instructions) == 9
            assert cfg.blocks[0].instructions == tuple(decoder.instructions[0:9])

            assert {cfg.blocks[1]} == cfg.edges[cfg.blocks[0]]

            # Block 1.
            assert len(cfg.blocks[1].instructions) == 6
            assert cfg.blocks[1].instructions == tuple(decoder.instructions[9:15])

            assert {cfg.blocks[2]} == cfg.edges[cfg.blocks[1]]

            # Block 2.
            assert len(cfg.blocks[2].instructions) == 8
            assert cfg.blocks[2].instructions == tuple(decoder.instructions[15:23])

            assert {cfg.blocks[3], cfg.blocks[4]} == cfg.edges[cfg.blocks[2]]

            # Block 3.
            assert len(cfg.blocks[3].instructions) == 2
            assert cfg.blocks[3].instructions == tuple(decoder.instructions[23:25])

            assert {cfg.blocks[4]} == cfg.edges[cfg.blocks[3]]

            # Block 4.
            assert len(cfg.blocks[4].instructions) == 1
            assert cfg.blocks[4].instructions == tuple(decoder.instructions[25:26])

            assert {cfg.blocks[5]} == cfg.edges[cfg.blocks[4]]

            # Block 5.
            assert len(cfg.blocks[5].instructions) == 2
            assert cfg.blocks[5].instructions == tuple(decoder.instructions[26:28])

            assert {cfg.blocks[6], cfg.blocks[12]} == cfg.edges[cfg.blocks[5]]

            # Block 6.
            assert len(cfg.blocks[6].instructions) == 3
            assert cfg.blocks[6].instructions == tuple(decoder.instructions[28:31])

            assert {cfg.blocks[7], cfg.blocks[12]} == cfg.edges[cfg.blocks[6]]

            # Block 7.
            assert len(cfg.blocks[7].instructions) == 2
            assert cfg.blocks[7].instructions == tuple(decoder.instructions[31:33])

            assert {cfg.blocks[8], cfg.blocks[11]} == cfg.edges[cfg.blocks[7]]

            # Block 8.
            assert len(cfg.blocks[8].instructions) == 3
            assert cfg.blocks[8].instructions == tuple(decoder.instructions[33:36])

            assert {cfg.blocks[9]} == cfg.edges[cfg.blocks[8]]

            # Block 9.
            assert len(cfg.blocks[9].instructions) == 4
            assert cfg.blocks[9].instructions == tuple(decoder.instructions[36:40])

            assert {cfg.blocks[9], cfg.blocks[10]} == cfg.edges[cfg.blocks[9]]

            # Block 10.
            assert len(cfg.blocks[10].instructions) == 1
            assert cfg.blocks[10].instructions == tuple(decoder.instructions[40:41])

            assert {cfg.blocks[12]} == cfg.edges[cfg.blocks[10]]

            # Block 11.
            assert len(cfg.blocks[11].instructions) == 4
            assert cfg.blocks[11].instructions == tuple(decoder.instructions[41:45])

            assert {cfg.blocks[12]} == cfg.edges[cfg.blocks[11]]

            # Block 12.
            assert len(cfg.blocks[12].instructions) == 1
            assert cfg.blocks[12].instructions == tuple(decoder.instructions[45:46])

            assert {cfg.blocks[13]} == cfg.edges[cfg.blocks[12]]

            # Block 13.
            assert len(cfg.blocks[13].instructions) == 6
            assert cfg.blocks[13].instructions == tuple(decoder.instructions[46:52])

            assert {cfg.blocks[2], cfg.blocks[14]} == cfg.edges[cfg.blocks[13]]

            # Block 14.
            assert len(cfg.blocks[14].instructions) == 1
            assert cfg.blocks[14].instructions == tuple(decoder.instructions[52:53])

            assert cfg.blocks[14] not in cfg.edges
