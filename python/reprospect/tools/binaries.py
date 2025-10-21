import abc
import dataclasses
import enum
import io
import logging
import pathlib
import re
import subprocess
import textwrap
import typing

import pandas
import rich.console
import rich.text
import rich.table
import typeguard

from reprospect.tools.architecture import NVIDIAArch

PATTERNS = [
    re.compile(r'-arch=sm_(\d+)'),
    re.compile(r'--gpu-architecture=compute_(\d+) --gpu-code=sm_(\d+)'),
    re.compile(r'--generate-code=arch=compute_(\d+),code=\[sm_(\d+)\]'),
    re.compile(r'--generate-code=arch=compute_(\d+),code=\[compute_(\d+),sm_(\d+)\]'),
    re.compile(r'--cuda-gpu-arch=sm_(\d+)'),
]

@typeguard.typechecked
def get_arch_from_compile_command(cmd : str) -> set[NVIDIAArch]:
    """
    Get `NVIDIA` architecture from compile command.
    """
    matches : set[str] = set()
    for pattern in PATTERNS:
        for match in pattern.finditer(cmd):
            matches.update(g for g in match.groups() if g)

    return {NVIDIAArch.from_compute_capability(cc = int(m)) for m in matches}

class DemanglerMixin(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_executable(cls) -> pathlib.Path | str:
        pass

    @classmethod
    @typeguard.typechecked
    def demangle(cls, s: str) -> str:
        """
        Demangle `s` (a symbol).
        """
        return subprocess.check_output([
            cls.get_executable(), s,
        ]).decode().strip()

class CuppFilt(DemanglerMixin):
    """
    Convenient wrapper for `cu++filt`.
    """
    @typing.override
    @classmethod
    @typeguard.typechecked
    def get_executable(cls) -> str:
        return 'cu++filt'

class LlvmCppFilt(DemanglerMixin):
    """
    Convenient wrapper for `llvm-cxxfilt`.
    """
    @typing.override
    @classmethod
    @typeguard.typechecked
    def get_executable(cls) -> str:
        return 'llvm-cxxfilt'

class ResourceUsage(enum.StrEnum):
    """
    Support for resource usage fields.

    Reference:
        - :cite:`nvidia-cuda-binary-utilities`
        - :cite:`openwall-wiki-parsing-nvidia`
        - :cite:`nvidia-cuda-compiler-driver-nvcc-statistics`
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
    @typeguard.typechecked
    def parse(line : str) -> dict:
        """
        Parse a resource usage line, such as produced by `cuobjdump` with `--dump-resource-usage`.
        """
        res = {}
        for token in re.findall(r'([A-Z]+)(?:\[([0-9]+)\])?:([0-9]+)', line):
            t = ResourceUsage(token[0])
            match t:
                case ResourceUsage.CONSTANT:
                    if t not in res:
                        res[t] = {}
                    res[t][int(token[1])] = int(token[2])
                case _:
                    res[t] = int(token[2])
        return res

@dataclasses.dataclass(slots = True)
class Function:
    code : str = None
    ru : dict = None

    @typeguard.typechecked
    def to_table(self, *, max_code_length: int = 130) -> rich.table.Table:
        """
        Convert to a :py:class:`rich.table.Table`.
        """
        table = rich.table.Table(width = 15 + max_code_length + 7, show_header = False, padding = (0, 1))
        table.add_column(width = 15)
        table.add_column(width = max_code_length, overflow = "ellipsis", no_wrap = True)

        # Code.
        table.add_row("Code", rich.text.Text(textwrap.dedent(self.code.expandtabs()).rstrip()), end_section = True)

        # Resource usage.
        table.add_row("Resource usage", ", ".join([f"{key}: {value}" for key, value in self.ru.items()]))

        return table

    @typeguard.typechecked
    def __str__(self) -> str:
        """
        Rich representation with :py:meth:`to_table`.
        """
        with rich.console.Console(width = 200) as console, console.capture() as capture:
            console.print(self.to_table(), no_wrap = True)
        return capture.get()

class CuObjDump:
    """
    Use `cuobjdump` for extracting `SASS`, symbol table, and so on.

    References:
        - :cite:`nvidia-cuda-binary-utility-cuobjdump`
    """
    @typeguard.typechecked
    def __init__(
        self,
        file : pathlib.Path,
        arch : NVIDIAArch,
        sass : bool = True,
        demangler : typing.Optional[typing.Type[CuppFilt | LlvmCppFilt]] = CuppFilt,
    ) -> None:
        self.file = file
        self.arch = arch
        self.functions = {}
        if sass:
            self.sass(demangler = demangler)

    @typeguard.typechecked
    def sass(self, demangler : typing.Optional[typing.Type[CuppFilt | LlvmCppFilt]] = None) -> None:
        """
        Extract `SASS` from `file`. Optionally demangle functions.
        """
        cmd = [
            'cuobjdump',
            '--gpu-architecture', self.arch.as_sm,
            '--dump-sass', '--dump-resource-usage',
            self.file,
        ]

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
    @typeguard.typechecked
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

        Note that extracting a `cubin` from `file` to extract a specific subset of the `SASS`
        instead of extracting the `SASS` straightforwardly from the whole `file` is
        significantly faster.
        """
        cmd = [
            'cuobjdump',
            '--extract-elf', cubin,
            '--gpu-architecture', arch.as_sm,
            file,
        ]

        logging.info(f'Extracting ELF file containing {cubin} from {file} for architecture {arch} with {cmd}.')

        files = subprocess.check_output(args = cmd, cwd = cwd).decode().splitlines()

        if len(files) != 1:
            raise RuntimeError(files)

        file = cwd / re.match(r'Extracting ELF file [ ]+ [0-9]+: ([A-Za-z0-9_.]+.cubin)', files[0]).group(1)

        return CuObjDump(file = file, arch = arch, **kwargs), file

    @staticmethod
    @typeguard.typechecked
    def symtab(*, cubin : pathlib.Path, arch : NVIDIAArch) -> pandas.DataFrame:
        """
        Extract the symbol table from `cubin` for `arch`.
        """
        cmd = ['cuobjdump', '--gpu-architecture', arch.as_sm, '--dump-elf', cubin]
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
