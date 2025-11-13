import logging
import pathlib
import subprocess
import typing

import pytest

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.binaries.elf import ELFHeader
from reprospect.utils              import cmake

from tests.python.compilation import get_compilation_output
from tests.python.cublas     import CuBLAS
from tests.python.parameters import Parameters, PARAMETERS

class TestELFHeader:
    CC_E_FLAGS : typing.Final[dict[int, int]] = {
        70  : 0b10001100000010101000110,     # 0x460546
        75  : 0b10010110000010101001011,     # 0x4b054b
        80  : 0b10100000000010101010000,     # 0x500550
        86  : 0b10101100000010101010110,     # 0x560556
        89  : 0b10110010000010101011001,     # 0x590559
        90  : 0b10110100000010101011010,     # 0x5a055a
        100 : 0b110000000000110010000000010, # 0x6006402
        103 : 0b110000000000110011100000010, # 0x6006702
        110 : 0b110000000000110111000000010, # 0x6006e02
        120 : 0b110000000000111100000000010, # 0x6007802
        121 : 0b110000000000111100100000010, # 0x6007902
    }
    """
    Values of `e_flags` obtained by calling ``cuobjdump --dump-elf`` on a cubin
    and reading the value of `flags` from the header for each architecture.
    """

    @pytest.mark.parametrize('cc,e_flags', CC_E_FLAGS.items())
    def test(self, cc: int, e_flags: int) -> None:
        decomposed = {
            'low': e_flags & 0xFF,
            'byte1': (e_flags >> 8) & 0xFF,
            'byte2': (e_flags >> 16) & 0xFF,
            'high': e_flags >> 24
        }
        # The compute capatbility is encoded in the low byte pre BLACKWELL,
        # and in byte1 (i.e., with an offset of 8) post BLACKWELL.
        if cc < 100:
            assert decomposed['low']  == cc
            assert decomposed['high'] == 0
        else:
            assert decomposed['byte1'] == cc
        logging.info(f'For cc {cc}, the e_flags {e_flags} are composed of {decomposed}.')

    @pytest.mark.parametrize('cc,e_flags', CC_E_FLAGS.items())
    def test_arch_from_e_flags(self, cc: int, e_flags: int) -> None:
        assert ELFHeader.arch_from_e_flags(e_flags) == NVIDIAArch.from_compute_capability(cc = cc)

class TestCUDART:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELFHeader` using the CUDA runtime shared library.
    """
    @pytest.fixture(scope = 'class')
    def cudart(self, cmake_file_api : cmake.FileAPI) -> pathlib.Path:
        cudart = pathlib.Path(cmake_file_api.cache['CUDA_CUDART']['value'])
        assert cudart.is_file()
        return cudart

    def test_shared_library(self, cudart : pathlib.Path) -> None:
        """
        The CUDA runtime shared library is a shared library and not itself a cubin.
        """
        descriptor = ELFHeader.decode(file = cudart)
        logging.info(descriptor)

        assert descriptor.e_type == 'shared'
        assert not descriptor.is_cuda

    def test_embedded_cubin(self, cudart : pathlib.Path) -> None:
        """
        The CUDA runtime shared library does not contain embedded cubins.
        """
        with pytest.raises(subprocess.CalledProcessError) as exc:
            tuple(CuObjDump.list_elf(file = cudart))
        assert 'does not contain device code' in exc.value.stderr

class TestCuBLAS:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELFHeader` using the cuBLAS shared library.
    """
    @pytest.fixture(scope = 'class')
    def cublas(self, cmake_file_api : cmake.FileAPI) -> CuBLAS:
        return CuBLAS(cmake_file_api = cmake_file_api)

    def test_shared_library(self, cublas : CuBLAS) -> None:
        """
        The cuBLAS shared library is a shared library and not itself a cubin.
        """
        descriptor = ELFHeader.decode(file = cublas.libcublas)
        logging.info(descriptor)

        assert descriptor.e_type == 'shared'
        assert not descriptor.is_cuda

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_embedded_cubin(self, parameters : Parameters, cublas : CuBLAS, workdir : pathlib.Path) -> None:
        """
        The cuBLAS shared library contains embedded cubins for many, but not all, architectures.
        """
        try:
            [cubin] = cublas.extract(arch = parameters.arch, cwd = workdir, randomly = True)
        except IndexError:
            pytest.skip(f"cuBLAS shared library does not contain an embedded cubin for arch {parameters.arch}")

        assert cubin.is_file()

        descriptor = ELFHeader.decode(file = cubin)
        logging.info(descriptor)

        assert descriptor.e_type == 'executable'
        assert descriptor.is_cuda
        assert descriptor.arch == parameters.arch

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
class TestSaxpy:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELFHeader` using an object file
    with a kernel that carries out a `saxpy`.
    """
    FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cu'

    @pytest.fixture(scope = 'function')
    def object_file(self, parameters : Parameters, cmake_file_api : cmake.FileAPI, workdir : pathlib.Path) -> pathlib.Path:
        """
        Compile into object file.
        """
        object_file, _ = get_compilation_output(
            source = self.FILE,
            cwd = workdir,
            arch = parameters.arch,
            object_file = True,
            resource_usage = True,
            cmake_file_api = cmake_file_api,
        )
        assert object_file.is_file()
        return object_file

    def test_object_file_and_embedded_cubin(self, parameters : Parameters, object_file : pathlib.Path, workdir : pathlib.Path) -> None:
        """
        Check that the object file:

        * is a relocatable and not itself a cubin.
        * contains an embedded cubin for the target architecture.
        """
        # object file
        descriptor = ELFHeader.decode(file = object_file)
        logging.info(descriptor)

        assert descriptor.e_type == 'relocatable'
        assert not descriptor.is_cuda

        # embedded cubin
        result = CuObjDump.extract_elf(file = object_file, arch = parameters.arch, name = 'saxpy', cwd = workdir)
        cubin = workdir / next(result)
        assert cubin.is_file()

        descriptor = ELFHeader.decode(file = cubin)
        logging.info(descriptor)

        assert descriptor.e_type == 'executable'
        assert descriptor.is_cuda
        assert descriptor.arch == parameters.arch
