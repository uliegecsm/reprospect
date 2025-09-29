import dataclasses
import logging
import pathlib
import re
import typing

import typeguard

@dataclasses.dataclass(frozen = True)
class Instruction:
    """
    Represents a single `SASS` instruction with its components.
    """
    offset : int
    instruction : str

class Decoder:
    """
    `NVIDIA` `SASS` instruction decoder.

    Simple decoder that parses `SASS` assembly and extracts key instruction information.
    """
    HEX = r'[0-9a-f]+'

    @typeguard.typechecked
    def __init__(self, source : typing.Optional[pathlib.Path] = None, code : typing.Optional[str] = None) -> None:
        """
        Initialize the decoder with the `SASS` contained in `source` or `code`.
        """
        self.source = source
        self.code   = code
        self.instructions: list[Instruction] = []
        self._parse()

    @typeguard.typechecked
    def _parse(self) -> None:
        """
        Parse `SASS` lines.
        """
        if self.source:
            lines = self.source.read_text().splitlines()
        else:
            lines = self.code.splitlines()

        iline = 0

        while iline < len(lines):
            line = lines[iline].strip()

            # Match instruction pattern:
            #   instruction
            instruction = re.match(
                rf'\/\*({self.HEX})\*\/\s+(.*)',
                line
            )

            if not instruction:
                logging.error(f'The line:\n\t{line}\ndid not match.')
                raise RuntimeError(line)

            # Extract instruction components.
            offset      = instruction.group(1).strip()
            instruction = instruction.group(2).strip()

            # Create instruction.
            instruction = Instruction(
                offset = int(offset, base = 16),
                instruction = instruction,
            )

            self.instructions.append(instruction)
            iline += 1
