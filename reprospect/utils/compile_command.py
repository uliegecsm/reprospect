import re
import typing

from reprospect.tools.architecture import NVIDIAArch

PATTERNS: typing.Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r'-arch=sm_(\d+)'),
    re.compile(r'--gpu-architecture=compute_(\d+) --gpu-code=sm_(\d+)'),
    re.compile(r'--generate-code=arch=compute_(\d+),code=\[sm_(\d+)\]'),
    re.compile(r'--generate-code=arch=compute_(\d+),code=\[compute_(\d+),sm_(\d+)\]'),
    re.compile(r'--cuda-gpu-arch=sm_(\d+)'),
)

def get_arch_from_compile_command(cmd: str) -> set[NVIDIAArch]:
    """
    Get NVIDIA architecture from compile command.

    >>> from reprospect.utils.compile_command import get_arch_from_compile_command
    >>> get_arch_from_compile_command('nvcc -arch=sm_89 test.cpp')
    {NVIDIAArch(family=<NVIDIAFamily.ADA: 'ADA'>, compute_capability=ComputeCapability(major=8, minor=9))}
    """
    matches: set[str] = set()
    for pattern in PATTERNS:
        for match in pattern.finditer(cmd):
            matches.update(g for g in match.groups() if g)

    return {NVIDIAArch.from_compute_capability(cc = int(m)) for m in matches}
