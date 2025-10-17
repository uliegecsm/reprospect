import dataclasses
import enum
import functools
import re
import typing

import typeguard

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

    @property
    @typeguard.typechecked
    def as_int(self) -> int:
        return self.major * 10 + self.minor

    def __str__(self) -> str:
        return f'{self.major}.{self.minor}'

    def __eq__(self, other: typing.Union[int, 'ComputeCapability']) -> bool:
        if isinstance(other, ComputeCapability):
            return (self.major, self.minor) == (other.major, other.minor)
        elif isinstance(other, int):
            return self.as_int == other
        else:
            return NotImplemented

    def __lt__(self, other : typing.Union[int, 'ComputeCapability']) -> bool:
        if isinstance(other, ComputeCapability):
            return (self.major, self.minor) < (other.major, other.minor)
        elif isinstance(other, int):
            return self.as_int < other
        else:
            return NotImplemented

    @staticmethod
    @typeguard.typechecked
    def from_int(value : int) -> 'ComputeCapability':
        major, minor = divmod(value, 10)
        return ComputeCapability(major = major, minor = minor)

class NVIDIAFamily(enum.StrEnum):
    """
    Supported `NVIDIA` architecture families.
    """
    VOLTA = 'VOLTA'
    TURING = 'TURING'
    AMPERE = 'AMPERE'
    ADA = 'ADA'
    HOPPER = 'HOPPER'
    BLACKWELL = 'BLACKWELL'

    @staticmethod
    @typeguard.typechecked
    def from_compute_capability(cc : ComputeCapability | int) -> "NVIDIAFamily":
        """
        Get the `NVIDIA` architecture family from a compute capability.

        See :cite:`nvidia-cuda-gpu-compute-capability`.
        """
        value = cc.as_int if isinstance(cc, ComputeCapability) else cc
        if value in [70, 72]:
            return NVIDIAFamily.VOLTA
        elif value == 75:
            return NVIDIAFamily.TURING
        elif value in [80, 86, 87]:
            return NVIDIAFamily.AMPERE
        elif value == 89:
            return NVIDIAFamily.ADA
        elif value == 90:
            return NVIDIAFamily.HOPPER
        elif value in [100, 103, 110, 120]:
            return NVIDIAFamily.BLACKWELL
        else:
            raise ValueError(f"unsupported compute capability {cc}")

@dataclasses.dataclass(frozen = True, eq = True, match_args = True, slots = True)
class NVIDIAArch:
    """
    `NVIDIA` architecture.

    It models `NVIDIA` GPU hardware identifiers — i.e., the microarchitecture family and compute capability.
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
    @typeguard.typechecked
    def as_compute(self) -> str:
        """
        Convert to `Cuda` "virtual architecture" (`compute_`).
        """
        return f'compute_{self.compute_capability.as_int}'

    @property
    @typeguard.typechecked
    def as_sm(self) -> str:
        """
        Convert to `Cuda` "real architecture" (`sm_`).
        """
        return f'sm_{self.compute_capability.as_int}'

    @typeguard.typechecked
    def __str__(self) -> str:
        return f"{self.family.name}{self.compute_capability.as_int}"

    @staticmethod
    @typeguard.typechecked
    def from_compute_capability(cc : str | int) -> 'NVIDIAArch':
        return NVIDIAArch(family = NVIDIAFamily.from_compute_capability(cc = cc), compute_capability = ComputeCapability.from_int(int(cc)))

    @staticmethod
    @typeguard.typechecked
    def from_str(arch : str) -> 'NVIDIAArch':
        family, cc = re.match(r'([A-Za-z]+)([0-9]+)', arch).groups()
        return NVIDIAArch(family = NVIDIAFamily(family.upper()), compute_capability = ComputeCapability.from_int(int(cc)))
