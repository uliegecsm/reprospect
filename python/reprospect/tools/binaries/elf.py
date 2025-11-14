"""
This module helps reading ELF headers to help detect if a file is a cubin.

References:

* https://gist.github.com/DtxdF/e6d940271e0efca7e0e2977723aec360
* :cite:`hayes-2019-decoding-cubin`
* https://en.wikipedia.org/wiki/Executable_and_Linkable_Format
* https://github.com/llvm/llvm-project/blob/46e9d6325a825b516826d0c56b6231abfaac16ab/llvm/include/llvm/BinaryFormat/ELF.h
* https://github.com/eliben/pyelftools
* https://refspecs.linuxfoundation.org/elf/elf.pdf
* https://uclibc.org/docs/elf-64-gen.pdf
"""
import dataclasses
import pathlib
import struct
import sys
import typing

from reprospect.tools.architecture import NVIDIAArch

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

EM_CUDA : typing.Final[int] = 190
"""
Value of :py:attr:`ELFHeader.e_machine` that identifies CUDA.

References:

* https://github.com/llvm/llvm-project/blob/46e9d6325a825b516826d0c56b6231abfaac16ab/llvm/include/llvm/BinaryFormat/ELF.h#L291
"""

ELFOSABI_CUDA : typing.Final[tuple[int, ...]] = (41, 51, 65)
"""
References:

* https://github.com/llvm/llvm-project/blob/46e9d6325a825b516826d0c56b6231abfaac16ab/llvm/include/llvm/BinaryFormat/ELF.h#L364-L365
"""

ELFABIVERSION_CUDA : typing.Final[tuple[int, int]] = (7, 8)
"""
References:

* https://github.com/llvm/llvm-project/blob/46e9d6325a825b516826d0c56b6231abfaac16ab/llvm/include/llvm/BinaryFormat/ELF.h#L391-L392
"""

EF_CUDA_SM_PRE_BLACKWELL: typing.Final[int] = 0xFF
"""
Mask for compute capability field pre BLACKWELL.

References:

* https://github.com/llvm/llvm-project/blob/12322b22c68a588caeee8702946695de0a8ba788/llvm/include/llvm/BinaryFormat/ELF.h#L932-L988
"""

EF_CUDA_SM_POST_BLACKWELL: typing.Final[int] = 0xFF00
"""
Mask for compute capability field post BLACKWELL.
"""

EF_CUDA_SM_OFFSET_POST_BLACKWELL: typing.Final[int] = 8
"""
Offset for compute capability field post BLACKWELL.
"""

class ELFHeaderEIdentClass(StrEnum):
    ELF_32 = '32-bit'
    ELF_64 = '64-bit'

    @classmethod
    def from_value(cls, value : int) -> 'ELFHeaderEIdentClass':
        """
        Create from ELF ``ei_class`` `value`.
        """
        match value:
            case 1: return cls.ELF_32
            case 2: return cls.ELF_64
            case _:
                raise ValueError(f'unsupported value {value}')

class Endianness(StrEnum):
    LITTLE = 'little'
    BIG = 'big'

    @classmethod
    def from_value(cls, value : int) -> 'Endianness':
        """
        Create from ELF ``ei_data`` `value`.
        """
        match value:
            case 1: return cls.LITTLE
            case 2: return cls.BIG
            case _:
                raise ValueError(f'unsupported value {value}')

@dataclasses.dataclass(frozen = True, slots = True)
class ELFHeaderEIdent:
    """
    ELF header `e_ident` field.
    """
    ei_class : ELFHeaderEIdentClass
    ei_data : Endianness
    ei_version : int
    ei_osabi : int
    ei_abiversion : int

    @classmethod
    def decode(cls, *, data : bytes) -> 'ELFHeaderEIdent':
        assert len(data) == 12
        return ELFHeaderEIdent(
            ei_class = ELFHeaderEIdentClass.from_value(data[0]),
            ei_data = Endianness.from_value(data[1]),
            ei_version = data[2],
            ei_osabi = data[3],
            ei_abiversion = data[4],
        )

class ELFHeaderEType(StrEnum):
    RELOCATABLE = 'relocatable'
    EXECUTABLE = 'executable'
    SHARED = 'shared'

    @classmethod
    def from_value(cls, value : int) -> 'ELFHeaderEType':
        match value:
            case 1: return cls.RELOCATABLE
            case 2: return cls.EXECUTABLE
            case 3: return cls.SHARED
            case _:
                raise ValueError(f'unsupported value {value}')

@dataclasses.dataclass(frozen = True, slots = True)
class ELFHeader:
    """
    ELF header.

    .. note:

        This class assumes the cubin to be of class 64-bit. CUDA 12.0 dropped support for 32-bit development.

    References:

    * https://nvidia.custhelp.com/app/answers/detail/a_id/5615/~/support-plan-for-32-bit-cuda
    """
    e_ident : ELFHeaderEIdent
    e_type : ELFHeaderEType #: ELF type.
    e_machine : int #: Target architecture.
    e_version : int #: ELF file version.
    e_flags : int #: Target-specific flags.

    @classmethod
    def decode(cls, *, file : pathlib.Path) -> 'ELFHeader':
        """
        Decode ELF file.
        """
        with file.open('rb') as fin:
            # ELF magic number.
            magic = fin.read(4)
            if magic != b'\x7fELF':
                raise RuntimeError(f'{file} misses the ELF magic number.')

            # e_ident fields (12 bytes after the 4 ELF magic number).
            e_ident = ELFHeaderEIdent.decode(data = fin.read(12))

            if e_ident.ei_class != ELFHeaderEIdentClass.ELF_64:
                raise RuntimeError(f"expecting cubin class to be 64-bit but got {e_ident.ei_class}")

            byte_order = '<' if e_ident.ei_data == Endianness.LITTLE else '>'

            # e_type (offset 0x10).
            e_type_bytes = fin.read(2)
            e_type = ELFHeaderEType.from_value(struct.unpack(byte_order + 'H', e_type_bytes)[0])

            # e_machine (offset 0x12).
            e_machine_bytes = fin.read(2)
            e_machine = struct.unpack(byte_order + 'H', e_machine_bytes)[0]

            # e_version (offset 0x14).
            e_version_bytes = fin.read(4)
            e_version = struct.unpack(byte_order + 'I', e_version_bytes)[0]

            # Skip e_entry, e_phoff and e_shoff.
            fin.seek(struct.calcsize('QQQ'), 1) # 3 x 8-byte unsigned long long

            # e_flags.
            e_flags_bytes = fin.read(4)
            e_flags = struct.unpack(byte_order + 'I', e_flags_bytes)[0]

            return ELFHeader(
                e_ident = e_ident,
                e_type = e_type,
                e_machine = e_machine,
                e_version = e_version,
                e_flags = e_flags,
            )

    @property
    def is_cuda(self) -> bool:
        """
        Return :py:obj:`True` if `file` is a valid CUDA binary file.
        """
        return self.e_machine == EM_CUDA \
            and self.e_ident.ei_osabi in ELFOSABI_CUDA \
            and self.e_ident.ei_abiversion in ELFABIVERSION_CUDA

def get_compute_capability_from_e_flags(value: int) -> int:
    """
    Return compute capability encoded in `e_flags`.
    """
    misc = (value >> 24) & 0xFF

    # Decode following the convention pre BLACKWELL.
    cc = value & EF_CUDA_SM_PRE_BLACKWELL

    # Bits 24-31 were zero pre BLACKWELL.
    if misc != 0 or cc not in range(70, 100):
        cc = (value & EF_CUDA_SM_POST_BLACKWELL) >> EF_CUDA_SM_OFFSET_POST_BLACKWELL

    return cc

def get_arch_from_elf_header(header : ELFHeader) -> NVIDIAArch:
    """
    Get compute capability encoded in `header` as NVIDIA architecture.
    """
    return NVIDIAArch.from_compute_capability(cc = get_compute_capability_from_e_flags(header.e_flags))
