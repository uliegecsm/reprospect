"""
Tools that can be used to extract SASS code and kernel attributes from executables.

.. note::

    Whereas :py:class:`cuda.core.experimental.ObjectCode` focuses on compiling with ``nvcc`` and querying a handful
    of kernel properties, this modules provides an interface to the CUDA binary analysis utilities to allow
    the SASS code of each kernel to be extracted for more extensive analysis, *e.g.* with :py:mod:`reprospect.tools.sass`.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
import re
import sys
import textwrap
import typing

import mypy_extensions
import pandas
import rich.console
import rich.text
import rich.table

from reprospect.tools.architecture       import NVIDIAArch
from reprospect.tools.binaries.demangle  import CuppFilt, LlvmCppFilt
from reprospect.tools.binaries.elf       import ELF
from reprospect.tools.binaries.symtab    import get_symbol_table
from reprospect.utils                    import rich_helpers
from reprospect.utils.subprocess_helpers import popen_stream

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

class ResourceType(StrEnum):
    """
    Resource usage fields.

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

class ResourceUsage(dict[ResourceType, int | dict[int, int]]):
    """
    Dictionary of resource usage.
    """
    @classmethod
    def parse(cls, line : str) -> ResourceUsage:
        """
        Parse a resource usage line, such as produced by ``cuobjdump`` with ``--dump-resource-usage``.
        """
        res : ResourceUsage = cls()
        for token in re.findall(r'([A-Z]+)(?:\[([0-9]+)\])?:([0-9]+)', line):
            t = ResourceType(token[0])
            match t:
                case ResourceType.CONSTANT:
                    if t not in res:
                        res[t] = {}
                    typing.cast(dict[int, int], res[t])[int(token[1])] = int(token[2])
                case _:
                    res[t] = int(token[2])
        return res

@mypy_extensions.mypyc_attr(native_class = True)
@dataclasses.dataclass(slots = True)
class Function:
    """
    Data structure holding the SASS code and resource usage of a kernel, as extracted from a binary file.
    """
    symbol : str #: The symbol name.
    code : str #: The SASS code.
    ru : ResourceUsage #: The resource usage.

    def to_table(self, *, max_code_length : int = 130, descriptors : dict[str, str] | None = None) -> rich.table.Table:
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

        # Symbol.
        table.add_row("Symbol", self.symbol, end_section = True)

        # Code.
        table.add_row("Code", rich.text.Text(textwrap.dedent(self.code.expandtabs()).rstrip()), end_section = True)

        # Resource usage.
        table.add_row("Resource usage", ", ".join(f"{key}: {value}" for key, value in self.ru.items()))

        return table

    def __str__(self) -> str:
        """
        Rich representation with :py:meth:`to_table`.
        """
        return rich_helpers.to_string(self.to_table())

@mypy_extensions.mypyc_attr(native_class = True)
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
        demangler : typing.Type[CuppFilt | LlvmCppFilt] | None = CuppFilt,
        keep : typing.Iterable[str] | None = None,
    ) -> None:
        """
        :param file: Either a host binary file containing one or more embedded CUDA binary files, or itself a CUDA binary file.
        :param keep: Optionally filter the functions to be kept.
        """
        self.file : typing.Final[pathlib.Path] = file #: The binary file.
        self.arch : typing.Final[NVIDIAArch] = arch #: The NVIDIA architecture.
        self.functions : dict[str, Function] = {}
        if sass:
            self.parse_sass(demangler = demangler, keep = keep)

    def parse_sass(self, demangler : typing.Type[CuppFilt | LlvmCppFilt] | None = None, keep : typing.Iterable[str] | None = None) -> None:
        """
        Parse SASS from :py:attr:`file`.
        """
        cmd : tuple[str | pathlib.Path, ...] = (
            'cuobjdump',
            '--gpu-architecture', self.arch.as_sm,
            '--dump-sass', '--dump-resource-usage',
            *((f'--function={",".join(keep)}',) if keep is not None else ()),
            self.file,
        )

        logging.info(f"Extracting 'SASS' from {self.file} using {cmd}.")

        START = '\t\tFunction : ' # pylint: disable=invalid-name
        STOP = '\t\t.....' # pylint: disable=invalid-name

        current_function_symbol_code : str | None = None
        current_function_code : str | None = None
        current_code : list[str] = []

        current_function_ru: str | None = None
        current_ru : dict[str, ResourceUsage] = {}

        # Read the stream as it comes.
        # The cuobjdump output starts with the resource usage for all functions.
        # Then, it gives the SASS code for each function.
        for line in popen_stream(args = cmd):

            # End of function block.
            if line.startswith(STOP):
                assert len(current_code) > 0 and current_function_symbol_code is not None and current_function_code is not None

                self.functions[current_function_code] = Function(
                    symbol = current_function_symbol_code,
                    code = ''.join(current_code),
                    ru = current_ru.pop(current_function_code),
                )

                current_function_code = None
                current_code.clear()

            # Inside a function block.
            elif current_function_code is not None:
                current_code.append(line)

            # Detect a function in the resource usage section.
            elif line.startswith(' Function '):
                mangled = line[10 : line.index(':', 10)]
                current_function_ru = demangler.demangle(mangled) if demangler else mangled

            # If previous line indicated resource usage.
            elif current_function_ru is not None:
                current_ru[current_function_ru] = ResourceUsage.parse(line = line)
                current_function_ru = None

            # Start of function block.
            elif line.startswith(START):
                assert len(current_code) == 0 and current_function_code is None

                current_function_symbol_code = line[len(START):].rstrip('\n')
                current_function_code = demangler.demangle(current_function_symbol_code) if demangler else current_function_symbol_code

            else:
                pass

    @classmethod
    def extract(cls, *,
        file : pathlib.Path,
        arch : NVIDIAArch,
        cwd : pathlib.Path,
        cubin : str,
        **kwargs,
    ) -> tuple[CuObjDump, pathlib.Path]:
        """
        Extract the embedded CUDA binary file whose name contains `cubin`,
        from `file`, for the given `arch`.

        The `file` can be inspected with the following command to list all `ELF` files::

            cuobjdump --list-elf <file>

        Note that extracting an embedded CUDA binary from a file so as to extract a specific
        subset of the SASS can be significantly faster than extracting all the SASS straightforwardly
        from the whole `file`.
        """
        files = tuple(cls.extract_elf(file = file, arch = arch, name = cubin, cwd = cwd))

        if len(files) != 1:
            raise RuntimeError(files)

        file = cwd / files[0]

        return CuObjDump(file = file, arch = arch, **kwargs), file

    @staticmethod
    def extract_elf(*,
        file : pathlib.Path,
        cwd : pathlib.Path | None = None,
        arch : NVIDIAArch | None = None,
        name : str | None = None,
    ) -> typing.Generator[str, None, None]:
        """
        Extract ELF files from `file`.

        :param arch: Optionally filter for a given architecture.
        :param name: Optionally filter by name.
        """
        cmd : list[str | pathlib.Path] = [
            'cuobjdump',
            '--extract-elf',
        ]

        if name is not None:
            cmd.append(name)

        if arch is not None:
            cmd.extend(('--gpu-architecture', arch.as_sm,))

        cmd.append(file)

        return (
            m.group(1)
            for f in popen_stream(args = cmd, cwd = cwd) if (m := re.match(r'Extracting ELF file [ ]+ [0-9]+: ([A-Za-z0-9_.]+\.cubin)', f)) is not None
        )

    def __str__(self) -> str:
        """
        Rich representation.
        """
        with rich.console.Console(width = 200) as console, console.capture() as capture:
            console.print(f'CuObjDump of {self.file} for architecture {self.arch}:')
            for name, function in self.functions.items():
                console.print(function.to_table(descriptors = {'Function' : name}), no_wrap = True)
        return capture.get()

    @functools.cached_property
    def embedded_cubins(self) -> tuple[str, ...]:
        """
        Get the names of the embedded CUDA binary files contained in :py:attr:`file`.
        """
        return tuple(self.list_elf(arch = self.arch, file = self.file))

    @staticmethod
    def list_elf(*, file : pathlib.Path, arch : NVIDIAArch | None = None) -> typing.Generator[str, None, None]:
        """
        List ELF files in `file`.

        :param arch: Optionally filter for a given architecture.
        """
        cmd : list[str | pathlib.Path] = [
            'cuobjdump',
            '--list-elf',
        ]

        if arch is not None:
            cmd.extend(('--gpu-architecture', arch.as_sm,))

        cmd.append(file)

        return (
            m.group(1)
            for f in popen_stream(args = cmd) if (m := re.match(r'ELF file [ ]+ [0-9]+: ([A-Za-z0-9_.]+\.cubin)', f)) is not None
        )

    @functools.cached_property
    def symtab(self) -> pandas.DataFrame:
        """
        Extract the symbol table from :py:attr:`file` for :py:attr:`arch`.

        This function requires that :py:attr:`file` is either a host binary file containing
        only a single embedded CUDA binary file or itself a CUDA binary file.
        """
        if not self.file_is_cubin:
            if len(self.embedded_cubins) != 1:
                raise RuntimeError('The host binary file contains more than one embedded CUDA binary file.')
        return get_symbol_table(file = self.file)

    @functools.cached_property
    def file_is_cubin(self) -> bool:
        """
        Whether :py:attr:`file` is a CUDA binary file.
        """
        with ELF(file = self.file) as elf:
            return elf.is_cuda
