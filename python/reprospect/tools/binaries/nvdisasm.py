import dataclasses
import logging
import pathlib
import re
import subprocess
import sys
import typing

from reprospect.tools.architecture       import NVIDIAArch
from reprospect.tools.binaries.cuobjdump import CuObjDump
from reprospect.tools.binaries.elf       import ELFHeader
from reprospect.tools.sass               import RegisterType

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

@dataclasses.dataclass(slots = True)
class NVDisasmFunction:
    """
    Data structure holding the SASS code and resource usage of a kernel, as extracted from a binary.
    :py:attr:`registers` holds detailed register usage information per register type. Each entry
    is a tuple holding:
    * the length of the span of used registers, *i.e.*, the maximum register index + 1
    * the number of registers actually used within that span
    For instance, if a kernel uses registers ``R0``, ``R1``, and ``R3``, then the entry for
    :py:const:`reprospect.tools.sass.RegisterType.GPR` will be `(4, 3)` because the span ``R0-R3``
    contains 4 registers, from which 3 are actually used.
    """
    disasm : str
    registers : dict[RegisterType, tuple[int, int]] | None = None

class NVDisasmRegisterState(StrEnum):
    """
    Register state, typically found in the output of ``nvdisasm``.
    References:
    * https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#nvdisasm
    """
    ASSIGNMENT             = '^'
    USAGE                  = 'v'
    USAGE_AND_REASSIGNMENT = 'x'
    IN_USE                 = ':'
    NOT_IN_USE             = ' '

    @property
    def used(self) -> bool:
        return self != self.NOT_IN_USE

class NVDisasm:
    """
    Extract information from CUDA binaries using ``nvdisasm``.
    ``nvdisasm`` can disassemble CUDA binary files and annotate the SASS assembly with
    information, such as register liveness range information. ``nvdisasm`` provides liveness
    ranges for all register types: ``GPR``, ``PRED``, ``UGPR``, ``UPRED``; see also
    :py:class:reprospect.tools.sass.RegisterType`.
    This class provides functionalities to parse this register liveness range information to
    deduce how many registers each kernel uses.
    Note that the register use information extracted by :py:class:`reprospect.tools.binaries.CuObjDump`
    concerns only the :py:const:`reprospect.tools.sass.RegisterType.GPR` register type. As compared with
    :py:class:`reprospect.tools.binaries.CuObjDump`, this class provides details for all register types.
    Note that register liveness range information can also be obtained by parsing the SASS
    code extracted by :py:class:`reprospect.tools.binaries.CuObjDump`. However, to implement
    such a parser, it is not sufficient to simply track the registers that appear in the SASS
    code. For instance, for certain instructions, operands span multiple consecutive registers,
    but only the first register index appears in the instruction string. For instance,
    * In ``STG.E desc[UR6][R6.64], R15``, the memory address operand ``[R6.64]`` uses two
      consecutive registers, namely, ``R6-R7``, but only ``R6`` appears explicitly.
    * In ``LDCU.64 UR8, c[0x0][0x3d8]``, the modifier ``64`` indicates that the destination is
      the two consecutive registers ``UR8-UR9``, but only ``UR8`` appears explicitly.
    * In ``IMAD.WIDE.U32 R2, R0, 0x4, R8``, the modifier ``WIDE`` indicates that ``R2`` and
      ``R8`` are twice as wide as ``R0`` and ``0x4``. Hence, the destination and the addend
      use ``R2-R3`` and ``R8-R9``, but only ``R2`` and ``R8`` appear explicitly.
    There are also complexities such as tracking register usage across function calls.
    Hence, this class relies on parsing the output of ``nvdisasm``, rather than on implementing
    its own parser.
    References:
    * https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#nvdisasm
    """
    Function      : typing.Final[typing.Type[NVDisasmFunction]]      = NVDisasmFunction      # pylint: disable=invalid-name
    RegisterState : typing.Final[typing.Type[NVDisasmRegisterState]] = NVDisasmRegisterState # pylint: disable=invalid-name

    HEADER_SEP  : typing.Final[re.Pattern[str]] = re.compile(r'^[ ]+\/\/ \+[\-\+]+\+$')
    HEADER_COLS : typing.Final[re.Pattern[str]] = re.compile(r'^[ ]+\/\/ \|(?:([A-Z ]+\|)+),?$')
    TABLE_CUT   : typing.Final[re.Pattern[str]] = re.compile(r'(?:\.[A-Za-z0-9_]+:)?[ ]+\/\/ \+[\.]+')
    TABLE_END   : typing.Final[re.Pattern[str]] = re.compile(r'(?:\.[A-Za-z0-9_]+:)?[ ]+\/\/ \+[\-\+]+\+$')

    def __init__(self, file : pathlib.Path, arch : typing.Optional[NVIDIAArch] = None) -> None:
        # Check that `file` is a CUDA binary file.
        descriptor = ELFHeader.decode(file = file)
        if not descriptor.is_cuda:
            raise ValueError(f'file {file} is not a CUDA binary file')

        # If the NVIDIA architecture is passed, check that `file` is a CUDA binary file for that arch.
        if arch:
            if descriptor.arch != arch:
                raise ValueError(f'file {file} is not a CUDA binary file for arch {arch}')
        else:
            arch = descriptor.arch

        self.file = file
        self.arch = arch
        self.cuobjdump = CuObjDump(file = self.file, arch = self.arch, sass = False)

        self.functions : dict[str, NVDisasmFunction] = {}

    def extract_register_usage(self, mangled : typing.Iterable[str]) -> None:
        """
        Extract the register usage information.
        ..note:
            We run ``nvdisasm`` separately for each function found.
            The reason we need to run it separately is that otherwise the "life range mode"
            adds to *all* kernels as many columns as needed by the most resource intensive kernel.
        """
        for function_mangled in mangled:
            # Disassemble the function with liveness range info.
            disasm = self._disassemble_with_liveness_range_info(file = self.file, function_mangled = function_mangled)

            # Parse the liveness range info.
            used = self._parse_liveness_range_info(disasm = disasm, function_mangled = function_mangled)

            # Extract how many registers are used.
            registers : dict[RegisterType, tuple[int, int]] = {
                reg_type: (len(used[reg_type]), sum(1 for is_used in used[reg_type] if is_used))
                for reg_type in used.keys()
            }

            self.functions[function_mangled] = self.Function(disasm = disasm, registers = registers)

    def _disassemble_with_liveness_range_info(self, *, file: pathlib.Path, function_mangled: str) -> str:
        """
        Disassemble a single function and store results.
        """
        # Find the function in the symbol table to determine its index.
        matches = self.cuobjdump.symtab[self.cuobjdump.symtab['name'] == function_mangled]
        if matches.shape[0] != 1:
            raise RuntimeError(f'The function {function_mangled} does not appear or appears more than once in the symbol table.')
        function_idx = matches.iloc[0]['index']

        # Run nvdisasm for the given function index.
        cmd : tuple[str | pathlib.Path, ...] = (
            'nvdisasm',
            '--life-range-mode=wide',
            '--separate-functions',
            '--print-code',
            '--cuda-function-index', function_idx,
            file,
        )
        logging.info(f'Disassembling function {function_mangled} from {file} using {cmd}.')

        return subprocess.check_output(args = cmd).decode()

    def _parse_liveness_range_info(self, *, disasm : str, function_mangled : str) -> dict:
        """
        Parse the liveness range info for a single function.
        """
        # Parse the disasm result.
        lines   = iter(disasm.splitlines())
        current = next(lines)

        # Find the line that looks like:
        #   //--------------------- .text.<mangled-function> --------------------------
        while not re.match(rf'\/\/[\-]+ \.text\.{function_mangled}[ ]+[\-]+', current):
            current = next(lines)

        # Skipping lines with:
        #   .section
        #   .sectionflags
        #   .sectioninfo
        #   .align
        current = next(lines)
        while re.match(pattern = r'^\t\.(section|sectionflags|sectioninfo|align)', string = current) is not None:
            current = next(lines)

        # Skip empty lines.
        while len(current.strip()) == 0:
            current = next(lines)

        # Let's process lines with the headers. They look like:
        #   // +--------------------+--------+----------------+
        #   // |    GPR             | PRED   |   UGPR         |
        #   // | # 0 1 2 3 4 5 6 7  |  # 0   | # 0 1 2 3 4 5  |
        #   // +--------------------+--------+----------------+
        if re.match(self.HEADER_SEP, current) is None:
            raise RuntimeError(f'unsupported format for {current}')
        current = next(lines)

        if re.match(self.HEADER_COLS, current) is None:
            raise RuntimeError(f'unsupported format for {current}')
        reg_types = [RegisterType[reg] for reg in re.findall(r'[A-Z]+', current)]
        current = next(lines)

        # Extract the register positions for each register type.
        start = current.find('// |')
        matched = re.match(pattern = r'^\/\/ \| (?:[0-9]+)?', string = current[start::])
        assert matched is not None
        sections = [x.strip() for x in current[start + matched.span()[1]-1::].strip().rstrip('|').split('|')]
        positions : dict[RegisterType, list[tuple[int, int]]] = {}
        for (reg_type, section) in zip(reg_types, sections):
            matches = re.finditer(r'\d+', section)
            assert matches is not None, f'No register positions found for {reg_type} in section "{section}"'
            positions[reg_type] = [matched.span() for matched in matches]

        # Parse the register usage for each line.
        used : dict[RegisterType, list[bool]] = {reg_type : [False] * len(positions[reg_type]) for reg_type in reg_types}

        current = next(lines)

        if re.match(self.HEADER_SEP, current) is None:
            raise RuntimeError(f'unsupported format for {current}')
        current = next(lines)

        while re.match(self.TABLE_END, current) is None:
            if re.match(self.TABLE_CUT, current) is not None:
                pass
            else:
                start = current.find('// |')
                matched = re.match(pattern = r'^\/\/ \| (?:[0-9]+)?', string = current[start::])
                assert matched is not None
                sections = [x.strip() for x in current[start + matched.span()[1]-1::].strip().rstrip('|').split('|')]
                for (reg_type, section) in zip(reg_types, sections):
                    offset = len(section) - len(section.lstrip('0123456789')) - 1
                    statuses_as_str = [
                        section[pos[0] + offset:pos[1] + offset].strip()
                        for pos in positions[reg_type]
                    ]
                    statuses = [self.RegisterState(s) if s else self.RegisterState.NOT_IN_USE for s in statuses_as_str]
                    used[reg_type] = [
                        already_used or status.used
                        for already_used, status in zip(used[reg_type], statuses)
                    ]
            current = next(lines)
        return used
