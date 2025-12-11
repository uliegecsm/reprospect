"""
This module provides helpers for architecture and CUDA version-dependent
features to enable feature-driven testing. It covers features that are:

* **Well-documented by NVIDIA**

  Official features with clear documentation, provided here for convenience.

* **Under-documented**

  Features mentioned in release notes or vendor communication but lacking comprehensive documentation.

* **Undocumented**

  Features discovered through empirical testing, profiling, or community knowledge.
"""

import dataclasses
import os

import semantic_version

from reprospect.tools.architecture import NVIDIAArch

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Memory:
    arch: NVIDIAArch
    version: semantic_version.Version = dataclasses.field(
        default_factory=lambda: semantic_version.Version(os.environ['CUDA_VERSION']),
    )

    @property
    def max_transaction_size(self) -> int:
        """
        Maximum memory transaction size in bytes for load/store operations.

        Prior to :py:attr:`reprospect.tools.architecture.NVIDIAFamily.BLACKWELL` and CUDA 13,
        a load/store of 32-byte aligned data requires two 16-byte transactions/instructions.

        Starting from :py:attr:`reprospect.tools.architecture.NVIDIAFamily.BLACKWELL` and CUDA 13,
        32-byte aligned data can be loaded/stored in a single transaction/instruction.

        >>> from semantic_version import Version
        >>> from reprospect.test.features import Memory
        >>> from reprospect.tools.architecture import NVIDIAArch
        >>> Memory(arch=NVIDIAArch.from_compute_capability(100), version=Version('13.0.0')).max_transaction_size
        32
        >>> Memory(arch=NVIDIAArch.from_compute_capability(90), version=Version('13.0.0')).max_transaction_size
        16

        References:

        * https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/#updated_vector_types
        """
        if self.arch.compute_capability < 100 or self.version in semantic_version.SimpleSpec('<13'):
            return 16
        return 32

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PTX:
    arch: NVIDIAArch

    @property
    def min_isa_version(self) -> semantic_version.Version: # pylint: disable=too-many-return-statements
        """
        Minimum PTX ISA version that supports :py:attr:`arch`.

        References:

        * https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes-ptx-release-history
        """
        match self.arch.compute_capability.as_int:
            case 70:
                return semantic_version.Version('6.0.0')
            case 75:
                return semantic_version.Version('6.3.0')
            case 80:
                return semantic_version.Version('7.0.0')
            case 86:
                return semantic_version.Version('7.1.0')
            case 87:
                return semantic_version.Version('7.4.0')
            case 89 | 90:
                return semantic_version.Version('7.8.0')
            case 100:
                return semantic_version.Version('8.6.0')
            case 103 | 121:
                return semantic_version.Version('8.8.0')
            case 110:
                return semantic_version.Version('9.0.0')
            case 120:
                return semantic_version.Version('8.7.0')
            case _:
                raise ValueError(self.arch)
