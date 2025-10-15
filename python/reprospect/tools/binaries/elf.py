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

import elftools.elf.elffile
import elftools.elf.sections

from reprospect.tools.architecture import NVIDIAArch, ComputeCapability

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

@dataclasses.dataclass(frozen = True, slots = True)
class TkInfo:
    """
    Specialized decoder for `.note.nv.tkinfo` section.

    Here is a typical section::

        .section .note.nv.tkinfo
        Entry: 1
            Owner                  Data size        Description
            NVIDIA Corp            140              NVIDIA CUDA Toolkit Information
                Note Version: 2
                Tool Name: ptxas
                Tool Version: Cuda compilation tools, release 13.0, V13.0.48
                Tool Branch: Build cuda_13.0.r13.0/compiler.36260728_0
                Tool Command Line Arguments: -arch sm_100 -m 64

    References:

    * https://github.com/tek-life/cuda-gdb/blob/080b643333b48c1c3aba2ba01f9fc7408c7bf6df/gdb/cuda/cuda-sass-json.h#L56-L62
    * https://github.com/tek-life/cuda-gdb/blob/080b643333b48c1c3aba2ba01f9fc7408c7bf6df/gdb/cuda/cuda-sass-json.c#L758-L768
    """
    note_version : int
    object_filename : str
    tool_name : str
    tool_version : str
    tool_branch : str
    tool_options : str

    @staticmethod
    def extract(arr : bytearray, offset : int) -> str:
        """
        The strings section starts at offset 24.
        """
        end = arr.find(b"\x00", 24 + offset)
        return arr[24 + offset:end].decode('ascii')

    @classmethod
    def decode(cls, *, note : elftools.elf.sections.NoteSection) -> typing.Generator['TkInfo', None, None]:
        """
        Iterate over the note entries and decode them.
        """
        if not note.elffile.little_endian:
            raise RuntimeError(note.elffile)

        for entry in note.iter_notes():
            (
                toolkit_version,
                object_filename,
                tool_name,
                tool_version,
                tool_branch,
                tool_options,
            ) = struct.unpack('<6I', entry['n_desc'][:24])

            yield TkInfo(
                note_version    = toolkit_version,
                object_filename = cls.extract(arr = entry['n_desc'], offset = object_filename),
                tool_name       = cls.extract(arr = entry['n_desc'], offset = tool_name),
                tool_version    = cls.extract(arr = entry['n_desc'], offset = tool_version),
                tool_branch     = cls.extract(arr = entry['n_desc'], offset = tool_branch),
                tool_options    = cls.extract(arr = entry['n_desc'], offset = tool_options),
            )

@dataclasses.dataclass(frozen = True, slots = True)
class CuInfo:
    """
    Specialized decoder for `.note.nv.cuinfo` section.

    Here is a typical section::

        .section .note.nv.cuinfo
        Entry: 1
            Owner                  Data size        Description
            NVIDIA Corp            8                NVIDIA CUDA Information
                Note Version: 2
                CUDA Virtual SM: sm_100
                CUDA Tool Kit Version: 13.0

    References:

    * https://github.com/tek-life/cuda-gdb/blob/080b643333b48c1c3aba2ba01f9fc7408c7bf6df/gdb/cuda/cuda-sass-json.h##L64-L65
    """
    note_version : int
    virtual_sm : int
    toolkit_version : int

    @classmethod
    def decode(cls, *, note : elftools.elf.sections.NoteSection) -> typing.Generator['CuInfo', None, None]:
        """
        Iterate the note entries and decode them.
        """
        if not note.elffile.little_endian:
            raise RuntimeError(note.elffile)

        for entry in note.iter_notes():
            if len(entry['n_desc']) != 8:
                raise RuntimeError(f'{entry!r} is not a valid .note.nv.cuinfo because it has too much description.')

            if entry.n_name != 'NVIDIA Corp':
                raise RuntimeError(f'{entry!r} is not a valid .note.nv.cuinfo.')

            yield CuInfo(
                note_version    = struct.unpack('<H', entry['n_desc'][0:2])[0],
                virtual_sm      = struct.unpack('<H', entry['n_desc'][2:4])[0],
                toolkit_version = struct.unpack('<I', entry['n_desc'][4:8])[0],
            )

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
        self.file : pathlib.Path = file #: Path to the ELF file.
        self.elf : elftools.elf.elffile.ELFFile | None = None

    def __enter__(self) -> Self:
        self.elf = elftools.elf.elffile.ELFFile(stream = self.file.open('rb'))
        return self

    def __exit__(self, *args, **kwargs) -> None:
        assert self.elf is not None
        self.elf.close()

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
