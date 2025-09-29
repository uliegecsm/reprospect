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
    hex : list[str]

class Decoder:
    """
    `NVIDIA` `SASS` instruction decoder.

    Simple decoder that parses `SASS` assembly and extracts key instruction information.
    """
    # Instruction is like:
    #   STG R1, R2
    INSTRUCTION = r'[A-Z0-9a-z,\. \[\]]+'

    # HEX is like:
    #   0x00000a0000017a02
    HEX = r'[a-f0-9x]+'

    # Typical SASS line:
    #   /*0070*/ MOV R5, 0x4 ; /* 0x0000000400057802 */
    MATCHER = rf'\/\*({HEX})\*\/\s+({INSTRUCTION})\s+;\s+\/\* ({HEX}) \*\/'

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

            match = re.match(self.MATCHER, line)

            if not match:
                logging.error(f'The line:\n\t{line}\ndid not match {self.MATCHER}.')
                raise RuntimeError(line)

            # Extract instruction components.
            offset      = match.group(1).strip()
            instruction = match.group(2).strip()
            hexes       = [match.group(3).strip()]

            # Check if next line contains the second hex word.
            if iline + 1 < len(lines):
                if (match := re.match(rf'\/\* ({self.HEX}) \*\/', lines[iline + 1].strip())):
                    hexes.append(match.group(1))
                    iline += 1

            # Create instruction.
            instruction = Instruction(
                offset = int(offset, base = 16),
                instruction = instruction,
                hex = hexes,
            )

            self.instructions.append(instruction)
            iline += 1
