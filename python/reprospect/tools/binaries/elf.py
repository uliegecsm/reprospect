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
import pathlib
import sys
import typing

import elftools.elf.elffile

from reprospect.tools.architecture import NVIDIAArch, ComputeCapability

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

class ELF:
    """
    Helper for reading ELF files and retrieve CUDA-specific information.
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

    def __init__(self, *, file : pathlib.Path) -> None:
        self.file = file #: Path to the ELF file.
        self.elf : elftools.elf.elffile.ELFFile | None = None

    def __enter__(self) -> Self:
        with self.file.open('rb') as fin:
            with elftools.elf.elffile.ELFFile(fin) as ready:
                self.elf = ready
                return self

    def __exit__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def is_cuda_impl(cls, *, header : elftools.construct.lib.container.Container) -> bool:
        return header['e_machine'] == 'EM_CUDA' \
            and header['e_ident']['EI_OSABI'] in cls.ELFOSABI_CUDA \
            and header['e_ident']['EI_ABIVERSION'] in cls.ELFABIVERSION_CUDA

    @property
    def is_cuda(self) -> bool:
        """
        Return :py:obj:`True` if :py:attr:`file` is a valid CUDA binary file.
        """
        return self.is_cuda_impl(header = self.header)

    @property
    def header(self) -> elftools.construct.lib.container.Container:
        assert self.elf is not None
        return self.elf.header

    @classmethod
    def compute_capability(cls, value) -> ComputeCapability:
        """
        Return compute capability encoded in `e_flags`.
        """
        misc = (value >> 24) & 0xFF

        # Decode following the convention pre BLACKWELL.
        cc = value & cls.EF_CUDA_SM_PRE_BLACKWELL

        # Bits 24-31 were zero pre BLACKWELL.
        if misc != 0 or cc not in range(70, 100):
            cc = (value & cls.EF_CUDA_SM_POST_BLACKWELL) >> cls.EF_CUDA_SM_OFFSET_POST_BLACKWELL

        return ComputeCapability.from_int(value = cc)

    @property
    def arch(self) -> NVIDIAArch:
        """
        Get compute capability encoded in `header` as NVIDIA architecture.
        """
        return NVIDIAArch.from_compute_capability(cc = self.compute_capability(value = self.header['e_flags']).as_int)
