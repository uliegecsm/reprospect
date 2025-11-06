import dataclasses
import functools
import re
import sys
import typing

import semantic_version

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

@functools.total_ordering
@dataclasses.dataclass(frozen = True, slots = True)
class ComputeCapability:
    """
    Compute capability.

    References:

    * https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability
    """
    major : int
    minor : int

    CUDA_SUPPORT : typing.ClassVar[dict[int, semantic_version.SimpleSpec]] = {
        70 : semantic_version.SimpleSpec('>=9,<=12.9'),
        75 : semantic_version.SimpleSpec('>=10'),
        80 : semantic_version.SimpleSpec('>=11.2'),
        86 : semantic_version.SimpleSpec('>=11.2'),
        87 : semantic_version.SimpleSpec('>=11.5'),
        89 : semantic_version.SimpleSpec('>=11.8'),
        90 : semantic_version.SimpleSpec('>=11.8'),
        100 : semantic_version.SimpleSpec('>=12.8'),
        103 : semantic_version.SimpleSpec('>=12.9'),
        110 : semantic_version.SimpleSpec('>=13.0'),
        120 : semantic_version.SimpleSpec('>=12.8'),
        121 : semantic_version.SimpleSpec('>=12.9'),
    }
    """
    CUDA Toolkit support.

    References:

    * https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
    * https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
    * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
    """

    @property
    def as_int(self) -> int:
        """
        >>> from reprospect.tools.architecture import ComputeCapability
        >>> ComputeCapability(major = 8, minor = 6).as_int
        86
        """
        return self.major * 10 + self.minor

    def __str__(self) -> str:
        return f'{self.major}.{self.minor}'

    def __eq__(self, other : object) -> bool:
        if isinstance(other, ComputeCapability):
            return (self.major, self.minor) == (other.major, other.minor)
        if isinstance(other, int):
            return self.as_int == other
        return NotImplemented

    def __lt__(self, other : typing.Union[int, 'ComputeCapability']) -> bool:
        if isinstance(other, ComputeCapability):
            return (self.major, self.minor) < (other.major, other.minor)
        if isinstance(other, int):
            return self.as_int < other
        return NotImplemented # type: ignore[unreachable]

    @staticmethod
    def from_int(value : int) -> 'ComputeCapability':
        """
        >>> from reprospect.tools.architecture import ComputeCapability
        >>> ComputeCapability.from_int(86)
        ComputeCapability(major=8, minor=6)
        """
        major, minor = divmod(value, 10)
        return ComputeCapability(major = major, minor = minor)

    def supported(self, version : semantic_version.Version) -> bool:
        """
        Check if the architecture is supported by the CUDA `version`.

        >>> from semantic_version import Version
        >>> from reprospect.tools.architecture import NVIDIAArch
        >>> NVIDIAArch.from_str('VOLTA70').compute_capability.supported(version = Version('13.0.0'))
        False
        """
        return version in self.CUDA_SUPPORT[self.as_int]

class NVIDIAFamily(StrEnum):
    """
    Supported NVIDIA architecture families.
    """
    VOLTA = 'VOLTA'
    TURING = 'TURING'
    AMPERE = 'AMPERE'
    ADA = 'ADA'
    HOPPER = 'HOPPER'
    BLACKWELL = 'BLACKWELL'

    @staticmethod
    def from_compute_capability(cc : ComputeCapability | int) -> "NVIDIAFamily":
        """
        Get the NVIDIA architecture family from a compute capability.

        See :cite:`nvidia-cuda-gpu-compute-capability`.
        """
        value = cc.as_int if isinstance(cc, ComputeCapability) else cc
        if value in [70, 72]:
            return NVIDIAFamily.VOLTA
        if value == 75:
            return NVIDIAFamily.TURING
        if value in [80, 86, 87]:
            return NVIDIAFamily.AMPERE
        if value == 89:
            return NVIDIAFamily.ADA
        if value == 90:
            return NVIDIAFamily.HOPPER
        if value in [100, 103, 110, 120]:
            return NVIDIAFamily.BLACKWELL
        raise ValueError(f"unsupported compute capability {cc}")

@dataclasses.dataclass(frozen = True, eq = True, match_args = True, slots = True)
class NVIDIAArch:
    """
    NVIDIA architecture.

    It models NVIDIA GPU hardware identifiers — i.e., the microarchitecture family and compute capability.
    """
    family : NVIDIAFamily
    compute_capability : ComputeCapability

    def __post_init__(self) -> None:
        """
        Validate that :py:attr:`compute_capability` matches :py:attr:`family`.
        """
        if NVIDIAFamily.from_compute_capability(self.compute_capability) != self.family:
            raise ValueError(
                f"Compute capability {self.compute_capability} does not match family {self.family.name}."
            )

    @property
    def as_compute(self) -> str:
        """
        Convert to CUDA "virtual architecture" (``compute_``).

        >>> from reprospect.tools.architecture import NVIDIAArch
        >>> NVIDIAArch.from_str('ADA89').as_compute
        'compute_89'
        """
        return f'compute_{self.compute_capability.as_int}'

    @property
    def as_sm(self) -> str:
        """
        Convert to CUDA "real architecture" (``sm_``).
        """
        return f'sm_{self.compute_capability.as_int}'

    def __str__(self) -> str:
        return f"{self.family.name}{self.compute_capability.as_int}"

    @staticmethod
    def from_compute_capability(cc : str | int) -> 'NVIDIAArch':
        return NVIDIAArch(family = NVIDIAFamily.from_compute_capability(cc = int(cc)), compute_capability = ComputeCapability.from_int(int(cc)))

    @staticmethod
    def from_str(arch : str) -> 'NVIDIAArch':
        """
        >>> from reprospect.tools.architecture import NVIDIAArch
        >>> NVIDIAArch.from_str('AMPERE86')
        NVIDIAArch(family=<NVIDIAFamily.AMPERE: 'AMPERE'>, compute_capability=ComputeCapability(major=8, minor=6))
        """
        if (matched := re.match(r'^([A-Za-z]+)([0-9]+)$', arch)) is None:
            raise ValueError(f'unsupported architecture {arch}')
        return NVIDIAArch(family = NVIDIAFamily(matched.group(1).upper()), compute_capability = ComputeCapability.from_int(int(matched.group(2))))
