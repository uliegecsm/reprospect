import dataclasses
import re
import typing

from reprospect.tools.sass.decode import Decoder, Instruction


@dataclasses.dataclass(frozen = True, slots = True)
class BasicBlock:
    """
    Basic block in the control flow graph.

    A basic block is a maximal sequence of consecutive instructions with:

    - single entry point (first instruction)
    - single exit point (last instruction)
    - no control flow instruction except possibly at the end
    """
    instructions: tuple[Instruction, ...]

    def __hash__(self) -> int:
        """
        A hash from the entry/exit instructions should be sufficiently unique (within a function).
        """
        return hash(self.instructions[0].offset + self.instructions[-1].offset)

@dataclasses.dataclass(slots = True)
class Graph:
    """
    Control Flow Graph (CFG) representation of SASS code.

    Blocks are :py:class:`reprospect.tools.sass.controlflow.BasicBlock`, edges represent control flow between blocks.
    """
    blocks: list[BasicBlock] = dataclasses.field(default_factory = list)

    edges: dict[BasicBlock, set[BasicBlock]] = dataclasses.field(default_factory = dict)
    """
    Adjacency list of where control flow may go for each block.
    """

    def add_block(self, block: BasicBlock) -> None:
        self.blocks.append(block)

    def add_blocks(self, blocks: typing.Iterable[BasicBlock]) -> None:
        self.blocks.extend(blocks)

    def add_edge(self, src: BasicBlock, dst: BasicBlock) -> None:
        self.edges.setdefault(src, set()).add(dst)

    def to_mermaid(self, title: str = 'CFG') -> str:
        """
        Generate a Mermaid flowchart diagram.

        References:

        * https://mermaid.js.org/syntax/flowchart.html
        """
        lines = [
            '---',
            f'title : {title}',
            '---',
            'flowchart TD',
        ]

        # Add blocks.
        for block in self.blocks:
            block_id = f'block_{block.instructions[0].offset}'
            block_text = '<br/>'.join(instr.instruction for instr in block.instructions)

            lines.append(f'\t{block_id}["{block_text}"]:::myblock')

        # Add edges.
        for src, dsts in self.edges.items():
            src_id = f'block_{src.instructions[0].offset}'
            for dst in dsts:
                dst_id = f'block_{dst.instructions[0].offset}'
                lines.append(f'\t{src_id} --> {dst_id}')

        # Style for all blocks.
        lines.append('\tclassDef myblock text-align:left')

        return '\n'.join(lines)

class ControlFlow:
    """
    Analyze SASS instructions from a :py:class:`reprospect.tools.sass.Decoder` to construct a :py:class:`reprospect.tools.sass.controlflow.Graph`.
    """
    BRANCHING: typing.Final[re.Pattern[str]] = re.compile('(' + '|'.join((
        'BPT',
        'BRA',
        'BREAK',
        'BRX',
        'BRXU',
        'EXIT',
        'JMP',
        'JMX',
        'JMXU',
        'RET',
    )) + ')')
    """
    Branching instructions that create a new basic block.
    """

    SYNCHRONIZATION: typing.Final[re.Pattern[str]] = re.compile('(' + '|'.join((
        'BSSY',
    )) + ')')

    @classmethod
    def analyze(cls, instructions: typing.Sequence[Instruction]) -> Graph:
        """
        Analyze instructions to discover blocks.
        """
        cfg = Graph()

        cfg.add_blocks(cls.create_blocks(instructions, cls.find_entry_points(instructions)))

        cls.add_control_flow_edges(cfg)

        return cfg

    @classmethod
    def get_target(cls, instruction: Instruction) -> int | None:
        """
        Get the target of a branching instruction.

        >>> from reprospect.tools.sass.controlflow import ControlFlow
        >>> from reprospect.tools.sass.decode      import ControlCode, Instruction
        >>> ControlFlow.get_target(instruction = Instruction(offset = '0160', instruction = f'@!P0 BRA {hex(400)}', hex = '0x0000000000088947', control = ControlCode.decode(code = '0x000fea0003800000')))
        400
        """
        if (target := re.search(rf'({Decoder.HEX})', instruction.instruction)) is not None:
            return int(target.group(1), 16)
        return None

    @classmethod
    def find_entry_points(cls, instructions: typing.Sequence[Instruction]) -> set[int]:
        """
        Identify entry point instructions (first instruction of each basic block).

        Entry points are:

        1. The first instruction.
        2. Any target of a branching or synchronization.
        3. Any instruction immediately following a branching.
        """
        # The first instruction is always an entry point.
        entry_points: set[int] = {instructions[0].offset}

        for iinstr, instr in enumerate(instructions):
            # Instruction after branch is an entry point.
            if cls.BRANCHING.search(instr.instruction) is not None:
                if iinstr + 1 < len(instructions):
                    entry_points.add(instructions[iinstr + 1].offset)

                # The instruction that the branch connects to is also an entry point.
                if (target := cls.get_target(instr)) is not None:
                    entry_points.add(target)

            elif cls.SYNCHRONIZATION.search(instr.instruction) is not None:
                target = cls.get_target(instr)
                assert target is not None
                entry_points.add(target)

        return entry_points

    @classmethod
    def create_blocks(cls, instructions: typing.Sequence[Instruction], entry_points: set[int]) -> tuple[BasicBlock, ...]:
        """
        Create basic blocks from entry point instructions.

        .. note::

            It discards blocks that are made of ``NOP`` instructions (as ``nvdisasm`` does).
        """
        blocks: list[BasicBlock] = []

        sorted_entry_points = sorted(entry_points)

        for iep, entry_point in enumerate(sorted_entry_points):
            # Determine end of this block.
            if iep + 1 < len(sorted_entry_points):
                next_entry_point = sorted_entry_points[iep + 1]
                block = BasicBlock(instructions = tuple(
                    instr for instr in instructions
                    if entry_point <= instr.offset < next_entry_point
                ))
            # Last block goes to end.
            else:
                block = BasicBlock(instructions = tuple(
                    instr for instr in instructions
                    if instr.offset >= entry_point
                ))

            if not all(instr.instruction == 'NOP' for instr in block.instructions):
                blocks.append(block)

        return tuple(blocks)

    @classmethod
    def add_control_flow_edges(cls, cfg: Graph) -> None:
        """
        Analyze control flow and add edges between blocks.
        """
        offset_to_block: typing.Final[dict[int, BasicBlock]] = {
            block.instructions[0].offset: block
            for block in cfg.blocks
        }

        for iblock, block in enumerate(cfg.blocks):
            add_fallthrough = True

            # Check if the last instruction is a branching instruction.
            if cls.BRANCHING.search(block.instructions[-1].instruction) is not None:
                # Add edge to branch target.
                if(target_offset := cls.get_target(block.instructions[-1])) is not None:
                    if target_offset in offset_to_block:
                        cfg.add_edge(block, offset_to_block[target_offset])

                # RET, EXIT, BRA etc. don't fall-through, except if there is an instruction predicate.
                if re.search(r'@!?U?P(?:T|\d+)', block.instructions[-1].instruction):
                    pass
                else:
                    add_fallthrough = False

            # Fall-through to next block.
            if add_fallthrough and iblock + 1 < len(cfg.blocks):
                cfg.add_edge(block, cfg.blocks[iblock + 1])
