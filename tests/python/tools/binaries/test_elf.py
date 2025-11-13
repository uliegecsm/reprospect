import logging
import pathlib
import subprocess
import typing

import pytest

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.binaries.elf import ELFHeader
from reprospect.utils              import cmake

from tests.python.cublas import CuBLAS

class TestELFHeader:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELFHeader`.
    """
    def test_cudart(self, cmake_file_api : cmake.FileAPI) -> None:
        """
        The CUDA runtime shared library is not a cubin and does not contain embedded cubins.
        """
        cudart = pathlib.Path(cmake_file_api.cache['CUDA_CUDART']['value'])
        assert cudart.is_file()

        descriptor = ELFHeader.decode(file = cudart)
        logging.info(descriptor)

        assert descriptor.e_type == 'shared'

        assert not descriptor.is_cuda

        with pytest.raises(subprocess.CalledProcessError) as exc:
            tuple(CuObjDump.list_elf(file = cudart))
        assert 'does not contain device code' in exc.value.stderr

    def test_cublas(self, cmake_file_api : cmake.FileAPI, workdir : pathlib.Path) -> None:
        """
        The cuBLAS shared library is not a cubin but it contains embedded cubins.
        """
        cublas = CuBLAS(cmake_file_api = cmake_file_api)

        descriptor = ELFHeader.decode(file = cublas.libcublas)
        logging.info(descriptor)

        assert descriptor.e_type == 'shared'

        assert not descriptor.is_cuda

        ARCH : typing.Final[NVIDIAArch] = NVIDIAArch.from_str('AMPERE80')

        [cubin] = cublas.extract(arch = ARCH, cwd = workdir, randomly = True)
        assert cubin.is_file()

        descriptor = ELFHeader.decode(file = cubin)
        logging.info(descriptor)

        assert descriptor.is_cuda

        assert descriptor.e_type == 'executable'
