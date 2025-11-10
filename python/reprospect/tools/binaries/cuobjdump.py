"""
Tools that can be used to extract SASS code and kernel attributes from executables.

.. note::

    Whereas :py:class:`cuda.core.experimental.ObjectCode` focuses on compiling with ``nvcc`` and querying a handful
    of kernel properties, this modules provides an interface to the CUDA binary analysis utilities to allow
    the SASS code of each kernel to be extracted for more extensive analysis, *e.g.* with :py:mod:`reprospect.tools.sass`.
"""

import dataclasses
import io
import logging
import pathlib
import re
import subprocess
import sys
import textwrap
import typing

import pandas
import rich.console
import rich.text
import rich.table

from reprospect.tools.architecture      import NVIDIAArch
from reprospect.tools.binaries.demangle import CuppFilt, LlvmCppFilt

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

ResourceUsageDict : typing.TypeAlias = dict['ResourceUsage', typing.Union[int, dict[int, int]]]

class ResourceUsage(StrEnum):
    """
    Support for resource usage fields.

    References:

    * :cite:`nvidia-cuda-binary-utilities`
    * :cite:`openwall-wiki-parsing-nvidia`
    * :cite:`nvidia-cuda-compiler-driver-nvcc-statistics`
    """
    # Registers.
    REGISTER = 'REG'
    # Shared memory.
    SHARED = 'SHARED'
    # Constant memory. Note that there might be several "banks" (cmem[0], cmem[1] and so on).
    CONSTANT = 'CONSTANT'

    LOCAL   = 'LOCAL'
    SAMPLER = 'SAMPLER'
    STACK   = 'STACK'
    SURFACE = 'SURFACE'
    TEXTURE = 'TEXTURE'

    @staticmethod
    def parse(line : str) -> ResourceUsageDict:
        """
        Parse a resource usage line, such as produced by ``cuobjdump`` with ``--dump-resource-usage``.
        """
        res : ResourceUsageDict = {}
        for token in re.findall(r'([A-Z]+)(?:\[([0-9]+)\])?:([0-9]+)', line):
            t = ResourceUsage(token[0])
            match t:
                case ResourceUsage.CONSTANT:
                    if t not in res:
                        res[t] = {}
                    typing.cast(dict[int, int], res[t])[int(token[1])] = int(token[2])
                case _:
                    res[t] = int(token[2])
        return res

@dataclasses.dataclass(slots = True)
class Function:
    """
    Data structure holding the SASS code and resource usage of a kernel, as extracted from a binary.
    """
    code : str #: The SASS code.
    ru : ResourceUsageDict | None = None #: The resource usage.

    def to_table(self, *, max_code_length : int = 130, descriptors : typing.Optional[dict[str, str]] = None) -> rich.table.Table:
        """
        Convert to a :py:class:`rich.table.Table`.

        :param descriptors: Key-value pairs added as descriptor rows at the top of the table, optional.
        """
        table = rich.table.Table(width = 15 + max_code_length + 7, show_header = False, padding = (0, 1))
        table.add_column(width = 15)
        table.add_column(width = max_code_length, overflow = "ellipsis", no_wrap = True)

        # Additional rows.
        if descriptors:
            for k, v in descriptors.items():
                table.add_row(k, v, end_section = True)

        # Code.
        table.add_row("Code", rich.text.Text(textwrap.dedent(self.code.expandtabs()).rstrip()), end_section = True)

        # Resource usage.
        if self.ru:
            table.add_row("Resource usage", ", ".join([f"{key}: {value}" for key, value in self.ru.items()]))

        return table

    def __str__(self) -> str:
        """
        Rich representation with :py:meth:`to_table`.
        """
        with rich.console.Console(width = 200) as console, console.capture() as capture:
            console.print(self.to_table(), no_wrap = True)
        return capture.get()

class CuObjDump:
    """
    Use ``cuobjdump`` for extracting SASS, symbol table, and so on.

    References:

    * :cite:`nvidia-cuda-binary-utility-cuobjdump`
    """
    def __init__(
        self,
        file : pathlib.Path,
        arch : NVIDIAArch,
        sass : bool = True,
        demangler : typing.Optional[typing.Type[CuppFilt | LlvmCppFilt]] = CuppFilt,
    ) -> None:
        self.file = file #: The binary file.
        self.arch = arch
        self.functions : dict[str, Function] = {}
        if sass:
            self.extract_sass(demangler = demangler)

    def extract_sass(self, demangler : typing.Optional[typing.Type[CuppFilt | LlvmCppFilt]] = None) -> None:
        """
        Extract SASS from :py:attr:`file`. Optionally demangle functions.
        """
        cmd : tuple[str | pathlib.Path, ...] = (
            'cuobjdump',
            '--gpu-architecture', self.arch.as_sm,
            '--dump-sass', '--dump-resource-usage',
            self.file,
        )

        logging.info(f"Extracting 'SASS' from {self.file} using {cmd}.")

        self.sass = subprocess.check_output(cmd).decode()

        assert self.sass is not None

        if demangler:
            for mangled in re.finditer(pattern = r'\t\tFunction : ([A-Za-z0-9_]+)', string = self.sass):
                self.sass = self.sass.replace(mangled.group(1), demangler.demangle(mangled.group(1)))

        # Retrieve SASS code for each function.
        START = '\t\tFunction : ' # pylint: disable=invalid-name
        STOP = '\t\t.....' # pylint: disable=invalid-name
        for match in re.finditer(rf'{START}(.*?){STOP}', self.sass, flags = re.DOTALL):
            function = match.group(1)
            name, code = function.split(sep = '\n', maxsplit = 1)
            self.functions[name] = Function(code = code)

        # Retrieve other function information.
        lines = iter(self.sass.splitlines())
        for line in lines:
            if line.startswith(' Function ') and line.endswith(':') and any(x in line for x in self.functions):
                function = line.replace(' Function ', '').rstrip(':')
                self.functions[function].ru = ResourceUsage.parse(line = next(lines))

    @staticmethod
    def extract(*,
        file : pathlib.Path,
        arch : NVIDIAArch,
        cwd : pathlib.Path,
        cubin : str,
        **kwargs,
    ) -> tuple['CuObjDump', pathlib.Path]:
        """
        Extract the `ELF` file whose name contains `cubin` from `file`, for the given `arch`.
        The `file` can be inspected with the following command to list all `ELF` files::

            cuobjdump --list-elf <file>

        Note that extracting a CUDA binary from a file to extract a specific subset of the SASS
        instead of extracting the SASS straightforwardly from the whole `file` is
        significantly faster.
        """
        cmd : tuple[str | pathlib.Path, ...] = (
            'cuobjdump',
            '--extract-elf', cubin,
            '--gpu-architecture', arch.as_sm,
            file,
        )

        logging.info(f'Extracting ELF file containing {cubin} from {file} for architecture {arch} with {cmd}.')

        files = subprocess.check_output(args = cmd, cwd = cwd).decode().splitlines()

        if len(files) != 1:
            raise RuntimeError(files)

        if (matched := re.match(r'Extracting ELF file [ ]+ [0-9]+: ([A-Za-z0-9_.]+.cubin)', files[0])) is not None:
            file = cwd / matched.group(1)
            return CuObjDump(file = file, arch = arch, **kwargs), file
        raise RuntimeError(files[0])

    def __str__(self) -> str:
        """
        Rich representation.
        """
        with rich.console.Console(width = 200) as console, console.capture() as capture:
            console.print(f'CuObjDump of {self.file} for architecture {self.arch}:')
            for name, function in self.functions.items():
                console.print(function.to_table(descriptors = {'Function' : name}), no_wrap = True)
        return capture.get()

    @staticmethod
    def symtab(*, cubin : pathlib.Path, arch : NVIDIAArch) -> pandas.DataFrame:
        """
        Extract the symbol table from `cubin` for `arch`.
        """
        cmd : list[str | pathlib.Path] = ['cuobjdump', '--gpu-architecture', arch.as_sm, '--dump-elf', cubin]
        logging.info(f'Extracting the symbol table from {cubin} using {cmd}.')

        # The section starts with
        #   .section .symtab
        # and ends with a blank line.
        output = io.StringIO()
        dump = False
        for line in subprocess.check_output(cmd).decode().splitlines():
            if line.startswith('.section .symtab'):
                dump = True
            elif dump and len(line.strip()) == 0:
                break
            elif dump:
                output.write(line + '\n')

        output.seek(0)

        return pandas.read_csv(output, sep = r'\s+')
