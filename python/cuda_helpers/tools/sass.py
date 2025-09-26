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
    line_number : int
    offset : str
    instruction : str
    target_registers : typing.Set[str]
    source_registers : typing.Set[str]
    control_code : str
    raw : str

class Decoder:
    """
    `NVIDIA` `SASS` instruction decoder.

    Simple decoder that parses `SASS` assembly and extracts key instruction information.
    """
    REGISTER_PATTERN = re.compile(r'\b(R\d+|UR\d+)\b')

    @typeguard.typechecked
    def __init__(self, source : pathlib.Path) -> None:
        """
        Initialize the decoder with the `SASS` contained in `source`.
        """
        self.source = source
        self.instructions: typing.List[Instruction] = []
        self._parse()

    @typeguard.typechecked
    def _parse(self) -> None:
        """
        Parse `SASS` lines.
        """
        lines = self.source.read_text().splitlines()

        headerflags = False

        i = 0
        line_number = 0

        while i < len(lines):
            line = lines[i].strip()

            # The line containing '..........' means the end of the SASS code.
            if '..........' in line:
                break

            # Skip empty lines.
            if not line:
                i += 1
                continue

            # Skip lines until '.headerflags' is met.
            if not headerflags and '.headerflags' not in line:
                i += 1
                continue

            # Skip '.headerflags' line.
            if '.headerflags' in line:
                headerflags = True
                i += 1
                continue

            # Match instruction pattern:
            #   /*offset*/ instruction ; /* hex1 */
            instruction = re.match(
                r'/\*([0-9a-fA-F]+)\*/\s+(.+?)\s*;\s*/\*\s*0x([0-9a-fA-F]+)\s*\*/',
                line
            )

            if not instruction:
                logging.error(f'The line:\n\t{line}\ndid not match.')
                raise RuntimeError(line)

            # Extract instruction components.
            offset = instruction.group(1)
            instruction_text = instruction.group(2).strip()
            hex1 = instruction.group(3)

            # Check if next line contains the second hex word (128-bit control code).
            control_code = [hex1]
            if i + 1 < len(lines):
                if (match := re.match(r'/\*\s*0x([0-9a-fA-F]+)\s*\*/', lines[i + 1].strip())):
                    hex2 = match.group(1)
                    control_code.append(hex2)
                    i += 1

            # Extract target and source registers.
            target_registers, source_registers = self._extract_registers(instruction_text)

            # Create instruction.
            instruction = Instruction(
                line_number = line_number,
                offset = offset,
                instruction = instruction_text,
                target_registers = target_registers,
                source_registers = source_registers,
                control_code = control_code,
                raw = line
            )

            self.instructions.append(instruction)
            line_number += 1
            i += 1

    @typeguard.typechecked
    def _extract_registers(self, instruction : str) -> typing.Tuple[typing.Set[str], typing.Set[str]]:
        """
        Extract target destination and source registers from instruction.

        TODO: Is it true that the target is always a single register ? If so, no need for a set.
        """
        target_registers = set()
        source_registers = set()

        # TODO: Find a better way to identify instructions that do no target any register.
        if ' ' not in instruction:
            return target_registers, source_registers

        # Retrieve the operation code and operands.
        opcode, operands = instruction.split(' ', 1)

        operands = [part.strip() for part in operands.split(',')]

        if not operands:
            return target_registers, source_registers

        # Store instruction look like:
        #   OPCODE ADDRESS SOURCE
        if opcode.upper().startswith(('STG', 'STL', 'STS', 'ST.')):
            address = self.REGISTER_PATTERN.findall(operands[0])
            source_registers.update(address)

            source = self.REGISTER_PATTERN.findall(operands[1])
            source_registers.update(source)

        # Predicates look like:
        #   OPCODE PX, PY, REG, REG, PT
        elif opcode.upper().startswith(('ISETP', 'FSETP', 'PSETP')):
            for operand in operands:
                register = self.REGISTER_PATTERN.findall(operand)
                source_registers.update(register)

        # Control flow instruction don't use any register.
        elif opcode.upper() in ('BRA', 'EXIT', 'NOP'):
            pass

        # Default case. The standard format looks like:
        #   OPCODE destination, source, source, ...
        else:
            destination = self.REGISTER_PATTERN.findall(operands[0])
            target_registers.update(destination)

            for operand in operands[1::]:
                source = self.REGISTER_PATTERN.findall(operand)
                source_registers.update(source)

        return target_registers, source_registers
