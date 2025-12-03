import logging
import pathlib
import re
import sys
import typing

import rich.table

from reprospect.tools.architecture       import NVIDIAArch
from reprospect.tools.binaries.demangle  import CuppFilt, LlvmCppFilt
from reprospect.tools.binaries.elf       import ELF
from reprospect.tools.binaries.symtab    import get_symbol_table
from reprospect.tools.sass.decode        import RegisterType
from reprospect.utils                    import rich_helpers
from reprospect.utils.subprocess_helpers import popen_stream

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Function(rich_helpers.TableMixin):
    """
    Data structure holding resource usage information of a kernel, as extracted from a binary.

    :py:attr:`registers` holds detailed register usage information per register type. Each entry
    is a tuple holding:

    * the length of the span of used registers, *i.e.*, the maximum register index + 1
    * the number of registers actually used within that span

    For instance, if a kernel uses registers ``R0``, ``R1``, and ``R3``, then the entry for
    :py:const:`reprospect.tools.sass.decode.RegisterType.GPR` will be `(4, 3)` because the span ``R0-R3``
    contains 4 registers, from which 3 are actually used.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('registers',)

    def __init__(self, registers : typing.Optional[dict[RegisterType, tuple[int, int]]] = None) -> None:
        self.registers : dict[RegisterType, tuple[int, int]] | None = registers

    @override
    def to_table(self) -> rich.table.Table:
        """
        Convert the register usage to a :py:class:`rich.table.Table`.
        """
        assert self.registers is not None

        rt = rich.table.Table()

        rt.add_column()
        rt.add_column("Span")
        rt.add_column("Used")

        for reg, info in self.registers.items():
            rt.add_row(
                reg.name,
                f'{reg}0-{reg}{info[0] - 1}' if info[0] > 1 else f'{reg}0',
                str(info[1])
            )

        return rt

class RegisterState(StrEnum):
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
        """
        Whether the state corresponds to a state in which the register is in use.
        """
        return self != self.NOT_IN_USE

class NVDisasm:
    """
    Extract information from CUDA binaries using ``nvdisasm``.

    The main purpose of ``nvdisasm`` is to disassemble CUDA binary files. Beyond the raw
    disassembly, it can also annotate the disassembled SASS with information, such as register
    liveness range information. ``nvdisasm`` provides liveness ranges for all register types:
    ``GPR``, ``PRED``, ``UGPR``, ``UPRED``; see also :py:class:`reprospect.tools.sass.decode.RegisterType`.

    This class provides functionalities to parse this register liveness range information to
    deduce how many registers each kernel uses.

    Note that the register use information extracted by :py:class:`reprospect.tools.binaries.CuObjDump`
    concerns only the :py:const:`reprospect.tools.sass.decode.RegisterType.GPR` register type. As compared with
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
    Consequently, to deduce the register usage, this class relies on parsing the register
    annotations provided by ``nvdisasm``, rather than on implementing its own logic to infer
    register usage from dumped SASS code.

    References:

    * https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#nvdisasm
    """
    HEADER_SEP      : typing.Final[re.Pattern[str]] = re.compile(r'^[ ]+\/\/ \+[\-\+]+\+$')
    HEADER_COLS     : typing.Final[re.Pattern[str]] = re.compile(r'^[ ]+\/\/ \|(?:([A-Z ]+\|)+),?$')
    TABLE_CUT       : typing.Final[re.Pattern[str]] = re.compile(r'(?:\.[A-Za-z0-9_]+:)?[ ]+\/\/ \+[\.]+')
    TABLE_BEGIN_END : typing.Final[re.Pattern[str]] = re.compile(r'(?:\.[A-Za-z0-9_]+:)?[ ]+\/\/ \+[\-\+]+\+$')

    def __init__(self,
        file : pathlib.Path,
        arch : typing.Optional[NVIDIAArch] = None,
        demangler : typing.Type[CuppFilt | LlvmCppFilt] = CuppFilt,
    ) -> None:
        """
        :param arch: Optionally check that `file` is a CUDA binary file for that `arch`.
        """
        with ELF(file = file) as elf:
            if not elf.is_cuda:
                raise ValueError(f'{file} is not a CUDA binary file')
            if arch is not None and elf.arch != arch:
                raise ValueError(f'{file} is not a CUDA binary file for {arch}')

        self.file = file
        self.arch = arch or elf.arch
        self.demangler = demangler

        self.symtab = get_symbol_table(file = self.file)

        self.functions : dict[str, Function] = {}

    def extract_register_usage_from_liveness_range_info(self, mangled : typing.Iterable[str]) -> None:
        """
        Extract register usage from liveness range information.

        .. note:

            The implementation runs ``nvdisasm`` separately for each function in `mangled`.
            Otherwise, the "life range mode" adds to *all* kernels as many columns as
            needed by the most resource intensive kernel.
        """
        for function_mangled in mangled:
            # Find the function in the symbol table to determine its index.
            matches = self.symtab[self.symtab['name'] == function_mangled]
            if matches.shape[0] != 1:
                raise RuntimeError(f'The function {function_mangled} does not appear or appears more than once in the symbol table.')
            function_idx = matches.iloc[0]['index']

            cmd : tuple[str | pathlib.Path, ...] = (
                'nvdisasm',
                '--life-range-mode=wide',
                '--separate-functions',
                '--print-code',
                '--cuda-function-index', str(function_idx),
                self.file,
            )

            logging.info(f'Disassembling function {function_mangled} from {self.file} using {cmd}.')

            self.functions[function_mangled] = self.parse_sass_with_liveness_range_info(function_mangled, popen_stream(args = cmd))

    @classmethod
    def parse_sass_with_liveness_range_info(cls, function_mangled : str, sass : typing.Iterator[str]) -> Function: # pylint: disable=too-many-branches
        """
        Parse the SASS with the liveness range information to extract the resource usage.

        It typically looks like::

            // +--------------------+--------+----------------+
            // |    GPR             | PRED   |   UGPR         |
            // | # 0 1 2 3 4 5 6 7  |  # 0   | # 0 1 2 3 4 5  |
            // +--------------------+--------+----------------+
            // |                    |        |                |
            // | 1   ^              |        |                |
            // | 2 ^ :              |        |                |
            // | 2 : :              |        | 1 ^            |
            // | 2 v :              |        | 1 :            |
            // +--------------------+--------+----------------+
        """
        found_function = False
        reg_types : typing.Optional[tuple[RegisterType, ...]] = None
        positions : typing.Optional[dict[RegisterType, tuple[tuple[int, int], ...]]] = None
        used : dict[RegisterType, tuple[bool, ...]] = {}

        for line in sass: # pylint: disable=too-many-nested-blocks
            # Find the line that looks like:
            #   //--------------------- .text.<mangled-function> --------------------------
            if not found_function:
                if re.match(rf'\/\/[\-]+ \.text\.{function_mangled}[ ]+[\-]+', line):
                    found_function = True
                else:
                    continue
            else:
                # Process lines with the headers.
                if reg_types is None:
                    if re.match(cls.HEADER_COLS, line) is not None:
                        reg_types = tuple(RegisterType[reg] for reg in re.findall(r'[A-Z]+', line))
                    # Skipping lines with:
                    #   .section
                    #   .sectionflags
                    #   .sectioninfo
                    #   .align
                    elif re.match(pattern = r'^\t\.(section|sectionflags|sectioninfo|align)', string = line) is not None:
                        continue
                    # Skip empty lines.
                    elif len(line.strip()) == 0:
                        continue
                    elif re.match(cls.HEADER_SEP, line) is not None:
                        continue
                    else:
                        raise RuntimeError('unexpected format')
                else:
                    if positions is None:
                        # Extract the register positions for each register type.
                        if (
                            (start := line.find('// |')) != -1
                            and (matched := re.match(pattern = r'^\/\/ \| (?:[0-9]+)?', string = line[start:])) is not None
                        ):
                            sections = tuple(x.strip() for x in line[start + matched.span()[1]-1::].strip().rstrip('|').split('|'))
                            positions = {}
                            for reg_type, section in zip(reg_types, sections):
                                matches = re.finditer(r'\d+', section)
                                if matches is None:
                                    raise RuntimeError(f'No register positions found for {reg_type} in section "{section}"')
                                positions[reg_type] = tuple(matched.span() for matched in matches)
                            used = {reg_type : (False,) * len(positions[reg_type]) for reg_type in reg_types}
                        else:
                            raise RuntimeError('unexpected format')
                    else:
                        # Parse the register usage for each line.
                        if (
                            (start := line.find('// |')) != -1
                            and (matched := re.match(pattern = r'^\/\/ \| (?:[0-9]+)?', string = line[start:])) is not None
                        ):
                            sections = tuple(x.strip() for x in line[start + matched.span()[1]-1::].strip().rstrip('|').split('|'))
                            for reg_type, section in zip(reg_types, sections):
                                offset = len(section) - len(section.lstrip('0123456789')) - 1
                                statuses_as_str = tuple(
                                    section[pos[0] + offset:pos[1] + offset].strip()
                                    for pos in positions[reg_type]
                                )
                                statuses = tuple(RegisterState(s) if s else RegisterState.NOT_IN_USE for s in statuses_as_str)
                                used[reg_type] = tuple(
                                    already_used or status.used
                                    for already_used, status in zip(used[reg_type], statuses)
                                )
                        elif re.match(cls.TABLE_CUT, line) is not None:
                            continue
                        elif re.match(cls.TABLE_BEGIN_END, line) is not None:
                            # If all register usages are False, we're beginning rather than ending, so we continue.
                            if not any(used[RegisterType.GPR]):
                                continue
                            break
                        else:
                            raise RuntimeError('unexpected format')

        # Extract how many registers are used.
        registers : dict[RegisterType, tuple[int, int]] = {
            reg_type : (len(vals), sum(vals)) for reg_type, vals in used.items()
        }

        return Function(registers = registers)

    def __str__(self) -> str:
        """
        Rich representation.
        """
        def to_table(name : str, function : Function) -> rich.table.Table:
            rt = rich.table.Table(show_header = False)
            rt.add_column(width = 130, overflow = "ellipsis", no_wrap = True)
            rt.add_row(self.demangler.demangle(name))
            rt.add_row(function.to_table())
            return rt

        return f'NVDisasm of {self.file} for architecture {self.arch}:\n' + ''.join(
            rich_helpers.to_string(to_table(name, func)) for name, func in self.functions.items()
        )
