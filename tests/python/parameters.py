import dataclasses
import functools
import os
import typing

import semantic_version

from reprospect.tools.architecture import NVIDIAArch

@dataclasses.dataclass(frozen = True, slots = True)
class Parameters:
    arch : NVIDIAArch

@functools.cache
def architectures(version : semantic_version.Version = semantic_version.Version(os.environ['CUDA_VERSION'])) -> tuple[NVIDIAArch, ...]:
    """
    Get the list of architectures to test, that are supported by `version`.
    """
    return tuple(
        arch for cc in [
            70,
            75,
            80,
            86,
            89,
            90,
            100,
            103,
            120,
        ] if (arch := NVIDIAArch.from_compute_capability(cc = cc)).compute_capability.supported(version = version)
    )

PARAMETERS : typing.Final[tuple[Parameters, ...]] = tuple(Parameters(arch = arch) for arch in architectures())
"""
Use this set of architectures when parametrizing a test, to ensure it covers as many relevant architectures as possible.
"""
