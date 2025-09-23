import dataclasses
import enum
import logging
import pathlib
import re
import subprocess

from cuda_helpers.tools.architecture import NVIDIAArch

import typeguard

PATTERNS = [
    re.compile(r'-arch=sm_(\d+)'),
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

    return {NVIDIAArch.from_compute_capability(m) for m in matches}

class CuppFilt:
    """
    Convenient wrapper for `cu++filt`.
    """
    @typeguard.typechecked
    @staticmethod
    def demangle(s: str) -> str:
        """
        Demangle `s` (a symbol).
        """
        return subprocess.check_output([
            'cu++filt', s
        ]).decode().strip()

class ResourceUsage(enum.StrEnum):
    """
    Support for resource usage fields.

    Reference:
        - https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
        - https://github.com/openwall/john/wiki/Parsing-nvidia-%27ptxas%27-output---memory-types
        - https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
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
                    if t not in res: res[t] = {}
                    res[t][int(token[1])] = int(token[2])
                case _:
                    res[t] = int(token[2])
        return res

class CuObjDump:
    """
    Use `cuobjdump` for extracting `SASS`, symbol table, and so on.

    References:
        - https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump
    """
    @dataclasses.dataclass
    class Function:
        code : str = None
        ru : dict = None

    @typeguard.typechecked
    def __init__(self, file : pathlib.Path, arch : NVIDIAArch, sass : bool = True) -> None:
        self.file = file
        self.arch = arch
        if sass:
            self.extract_and_parse_sass(demangle = True)

    @typeguard.typechecked
    def extract_and_parse_sass(self, demangle : bool) -> None:
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

        if demangle:
            for mangled in re.finditer(pattern = r'\t\tFunction : ([A-Za-z0-9_]+)', string = self.sass):
                self.sass = self.sass.replace(mangled.group(1), CuppFilt.demangle(mangled.group(1)))

        self.functions = {}

        # Retrieve SASS code for each function.
        START = '\t\tFunction : '
        STOP = '\t\t.....'
        for match in re.finditer(rf'{START}(.*?){STOP}', self.sass, flags = re.DOTALL):
            function = match.group(1)
            name, code = function.split(sep = '\n', maxsplit = 1)
            self.functions[name] = CuObjDump.Function(code = code)

        # Retrieve other function information.
        lines = iter(self.sass.splitlines())
        for line in lines:
            if line.startswith(' Function ') and any(x in line for x in self.functions.keys()) and line.endswith(':'):
                function = line.replace(' Function ', '').rstrip(':')
                self.functions[function].ru = ResourceUsage.parse(line = next(lines))
