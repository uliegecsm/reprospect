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
import textwrap
import typing

import mypy_extensions
import pandas
import regex
import rich.console
import rich.text
import rich.table

from reprospect.tools.architecture       import NVIDIAArch
from reprospect.tools.binaries.demangle  import CuppFilt, LlvmCppFilt
from reprospect.tools.binaries.elf       import ELF
from reprospect.tools.binaries.symtab    import get_symbol_table
from reprospect.utils                    import rich_helpers
from reprospect.utils.subprocess_helpers import popen_stream

RESOURCE_USAGE_PATTERN: typing.Final[regex.Pattern[str]] = regex.compile(
    r'\s*'
    r'REG:(?P<register>\d+)\s+'
    r'STACK:(?P<stack>\d+)\s+'
    r'SHARED:(?P<shared>\d+)\s+'
    r'LOCAL:(?P<local>\d+)\s+'
    r'(CONSTANT\[(?P<constant_bank>\d+)\]:(?P<constant_size>\d+)\s*)+'
    r'TEXTURE:(?P<texture>\d+)\s+'
    r'SURFACE:(?P<surface>\d+)\s+'
    r'SAMPLER:(?P<sampler>\d+)',
)

@mypy_extensions.mypyc_attr(native_class=True)
@dataclasses.dataclass(slots=True, frozen=True, repr=True)
class ResourceUsage: # pylint: disable=too-many-instance-attributes
    """
    Resource usage.

    References:

    * :cite:`nvidia-cuda-binary-utilities`
    * :cite:`openwall-wiki-parsing-nvidia`
    * :cite:`nvidia-cuda-compiler-driver-nvcc-statistics`
    """
    register: int = 0
    constant: dict[int, int] = dataclasses.field(default_factory=dict)
    shared: int = 0
    local: int = 0
    sampler: int = 0
    stack: int = 0
    surface: int = 0
    texture: int = 0

    def __str__(self) -> str:
        return f'REG: {self.register}, STACK: {self.stack}, SHARED: {self.shared}, LOCAL: {self.local}, CONSTANT: {self.constant}, TEXTURE: {self.texture}, SURFACE: {self.surface}, SAMPLER: {self.sampler}'

    @classmethod
    def parse(cls, line : str) -> ResourceUsage:
        """
        Parse a resource usage line, such as produced by ``cuobjdump`` with ``--dump-resource-usage``.
        """
        if not (matched := RESOURCE_USAGE_PATTERN.match(line)):
            raise ValueError(line)

        return cls(
            register=int(matched.group('register')),
            stack=int(matched.group('stack')),
            shared=int(matched.group('shared')),
            local=int(matched.group('local')),
            constant={int(bank): int(value) for bank, value in zip(matched.captures('constant_bank'), matched.captures('constant_size'), strict=True)},
            texture=int(matched.group('texture')),
            surface=int(matched.group('surface')),
            sampler=int(matched.group('sampler')),
        )

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
        table.add_row("Resource usage", str(self.ru))

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
        *,
        sass : bool = True,
        demangler : type[CuppFilt | LlvmCppFilt] | None = CuppFilt,
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

    def parse_sass(self, demangler : type[CuppFilt | LlvmCppFilt] | None = None, keep : typing.Iterable[str] | None = None) -> None:
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
            cmd.extend(('--gpu-architecture', arch.as_sm))

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
            cmd.extend(('--gpu-architecture', arch.as_sm))

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
