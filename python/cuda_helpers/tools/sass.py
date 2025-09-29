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
    instruction : str

class Decoder:
    """
    `NVIDIA` `SASS` instruction decoder.

    Simple decoder that parses `SASS` assembly and extracts key instruction information.
    """
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
                r'(.*)',
                line
            )

            if not instruction:
                logging.error(f'The line:\n\t{line}\ndid not match.')
                raise RuntimeError(line)

            # Extract instruction components.
            instruction = instruction.group(1).strip()

            # Create instruction.
            instruction = Instruction(
                instruction = instruction,
            )

            self.instructions.append(instruction)
            iline += 1
