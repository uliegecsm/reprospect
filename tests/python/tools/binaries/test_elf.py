import logging
import os
import pathlib
import subprocess
import typing

import pytest

from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.binaries.elf import ELFHeader
from reprospect.utils              import cmake

@pytest.fixture(scope = 'session')
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

@pytest.fixture(scope = 'session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

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
        cublas = pathlib.Path(cmake_file_api.cache['CUDA_CUDART']['value']).parent / 'libcublas.so'
        assert cublas.is_file()

        descriptor = ELFHeader.decode(file = cublas)
        logging.info(descriptor)

        assert descriptor.e_type == 'shared'

        assert not descriptor.is_cuda

        NAME : typing.Final[str] = 'libcublas.6.sm_80.cubin'

        subprocess.check_call((
            'cuobjdump',
            '--extract-elf', NAME,
            '--gpu-architecture', 'sm_80',
            cublas,
        ), cwd = workdir)

        cubin = workdir / NAME
        assert cubin.is_file()

        descriptor = ELFHeader.decode(file = cubin)
        logging.info(descriptor)

        assert descriptor.is_cuda

        assert descriptor.e_type == 'executable'
