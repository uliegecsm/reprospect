import re

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
