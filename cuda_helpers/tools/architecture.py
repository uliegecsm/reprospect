import dataclasses
import enum
import re

import typeguard

class NVIDIAFamily(enum.StrEnum):
    """
    Supported `NVIDIA` architecture families.
    """
    VOLTA = 'VOLTA'
    AMPERE = 'AMPERE'
    HOPPER = 'HOPPER'
    BLACKWELL = 'BLACKWELL'

    @staticmethod
    @typeguard.typechecked
    def from_compute_capability(cc: str | int) -> "NVIDIAFamily":
        cc = int(cc)
        if 70 <= cc < 80:
            return NVIDIAFamily.VOLTA
        elif 80 <= cc < 90:
            return NVIDIAFamily.AMPERE
        elif 90 <= cc < 100:
            return NVIDIAFamily.HOPPER
        elif 100 <= cc < 130:
            return NVIDIAFamily.BLACKWELL
        else:
            raise ValueError(f"unsupported compute capability {cc}")

@dataclasses.dataclass(frozen = True, eq = True, match_args = True)
class NVIDIAArch:
    """
    `NVIDIA` architecture.

    It models `NVIDIA` GPU hardware identifiers — i.e., the microarchitecture family and compute capability.
    """
    family : NVIDIAFamily
    compute_capability : int

    @property
    @typeguard.typechecked
    def as_compute(self) -> str:
        """
        Convert to `Cuda` "virtual architecture" (`compute_`).
        """
        return f'compute_{self.compute_capability}'

    @property
    @typeguard.typechecked
    def as_sm(self) -> str:
        """
        Convert to `Cuda` "real architecture" (`sm_`).
        """
        return f'sm_{self.compute_capability}'

    @typeguard.typechecked
    def __str__(self) -> str:
        return f"{self.family.name}{self.compute_capability}"

    @staticmethod
    @typeguard.typechecked
    def from_compute_capability(cc : str | int) -> 'NVIDIAArch':
        return NVIDIAArch(family = NVIDIAFamily.from_compute_capability(cc = cc), compute_capability = int(cc))

    @staticmethod
    @typeguard.typechecked
    def from_str(arch : str) -> 'NVIDIAArch':
        family, cc = re.match(r'([A-Za-z]+)([0-9]+)', arch).groups()
        return NVIDIAArch(family = NVIDIAFamily(family.upper()), compute_capability = int(cc))
