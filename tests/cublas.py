"""
Using real-world code improves tests.
This module helps retrieve code from cuBLAS.
"""

import pathlib
import random
import typing

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.utils              import cmake

class CuBLAS:
    CUDA_CUDART: typing.Final[str] = 'CUDA_CUDART'

    def __init__(self, cmake_file_api: cmake.FileAPI) -> None:
        """
        Find the cuBLAS shared library, assuming that the CMake cache populates :py:attr:`CUDA_CUDART`.
        """
        self.libcublas = pathlib.Path(cmake_file_api.cache[self.CUDA_CUDART]['value']).parent / 'libcublas.so' #: Shared library path.
        assert self.libcublas.is_file()

    def extract(self, *, arch: NVIDIAArch, cwd: pathlib.Path, randomly: bool = True) -> tuple[pathlib.Path, ...]:
        """
        Extract CUDA binary files from :py:attr:`libcublas`.

        :param cwd: Where to extract the files.
        :param randomly: If :py:obj:`True`, extract only one file (randomly). Otherwise, extract *all* files.

        :raises IndexError: If `randomly` is :py:obj:`True` but there is no embedded cubin for `arch`.
        """
        name: typing.Final[str | None] = random.choice(
            tuple(CuObjDump.list_elf(arch = arch, file = self.libcublas)),
        ) if randomly else None

        return tuple(
            cwd / file
            for file in CuObjDump.extract_elf(file = self.libcublas, arch = arch, name = name, cwd = cwd)
        )
