import logging
import os
import pathlib
import re
import subprocess
import typing

import elftools.elf.sections
import pytest
import semantic_version

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.binaries.elf import ELF, TkInfo, CuInfo
from reprospect.utils              import cmake, nvcc

from tests.python.compilation import get_compilation_output
from tests.python.cublas      import CuBLAS
from tests.python.parameters  import Parameters, PARAMETERS

class TestGetComputeCapabilityFromEFlags:
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
    def test(self, cc : int, e_flags : int) -> None:
        b0 = e_flags         & 0xFF
        b1 = (e_flags >> 8)  & 0xFF
        b2 = (e_flags >> 16) & 0xFF
        b3 =  e_flags >> 24

        # The compute capatbility is encoded in the low byte pre BLACKWELL,
        # and in byte 1 (i.e., with an offset of 8) post BLACKWELL.
        if cc < 100:
            assert b0  == cc and b3 == 0
        else:
            assert b1 == cc

        logging.info(f'For cc {cc}, the e_flags {e_flags} are composed of {b0}, {b1}, {b2} and {b3}.')

    @pytest.mark.parametrize('cc,e_flags', CC_E_FLAGS.items())
    def test_get_compute_capability_from_e_flags(self, cc : int, e_flags : int) -> None:
        assert ELF.compute_capability(e_flags).as_int == cc

class TestCUDART:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELF` using the CUDA runtime shared library.
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
        with ELF(file = cudart) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_DYN'
            assert not elf.is_cuda

    def test_embedded_cubin(self, cudart : pathlib.Path) -> None:
        """
        The CUDA runtime shared library does not contain embedded cubins.
        """
        with pytest.raises(subprocess.CalledProcessError) as exc:
            tuple(CuObjDump.list_elf(file = cudart))
        assert 'does not contain device code' in exc.value.stderr

def get_cubin(arch : NVIDIAArch, cublas : CuBLAS, workdir : pathlib.Path) -> pathlib.Path:
    try:
        [cubin] = cublas.extract(arch = arch, cwd = workdir, randomly = True)
    except IndexError:
        pytest.skip(f'{cublas} does not contain an embedded cubin for {arch}.')
    assert cubin.is_file()
    return cubin

def get_cuinfo_and_tkinfo(*, arch : NVIDIAArch, file : pathlib.Path, version : semantic_version.Version = semantic_version.Version(os.environ['CUDA_VERSION'])) -> tuple[CuInfo | None, TkInfo | None]:
    """
    Extract `cuinfo` and `tkinfo` note sections.

    It takes care of checking if each note section has to exist or not, given the `arch` and CUDA `version`.
    """
    with ELF(file = file) as elf:

        def has(*, name : str) -> bool:
            return elf.elf.has_section(section_name = '.note.nv.' + name)

        def get(*, name : str) -> elftools.elf.sections.NoteSection:
            section = elf.elf.get_section_by_name(name = '.note.nv.' + name)
            assert isinstance(section, elftools.elf.sections.NoteSection)
            return section

        cuinfos : tuple[CuInfo, ...] | None = tuple(note for note in CuInfo.decode(note = get(name = 'cuinfo'))) if has(name = 'cuinfo') else None
        tkinfos : tuple[TkInfo, ...] | None = tuple(note for note in TkInfo.decode(note = get(name = 'tkinfo'))) if has(name = 'tkinfo') else None

        cuinfo : CuInfo | None = None
        if version in semantic_version.SimpleSpec('<13.0.0'):
            assert cuinfos is None
        else:
            logging.info(cuinfos)
            assert cuinfos is not None
            assert len(cuinfos) == 1
            cuinfo = cuinfos[0]
            assert cuinfo.virtual_sm == arch.compute_capability.as_int

        tkinfo : TkInfo | None = None
        if version in semantic_version.SimpleSpec('<13.0.0') \
            and arch.compute_capability <= 90:
            assert tkinfos is None
        else:
            logging.info(tkinfos)
            assert tkinfos is not None
            assert len(tkinfos) == 1
            tkinfo = tkinfos[0]

        if cuinfo is not None and tkinfo is not None:
            assert cuinfo.note_version == tkinfo.note_version

        return cuinfo, tkinfo

class TestCuBLAS:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELF` using the cuBLAS shared library.
    """
    @pytest.fixture(scope = 'class')
    def cublas(self, cmake_file_api : cmake.FileAPI) -> CuBLAS:
        return CuBLAS(cmake_file_api = cmake_file_api)

    def test_shared_library(self, cublas : CuBLAS) -> None:
        """
        The cuBLAS shared library is a shared library and not itself a cubin.
        """
        with ELF(file = cublas.libcublas) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_DYN'
            assert not elf.is_cuda

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_embedded_cubin(self, parameters : Parameters, cublas : CuBLAS, workdir : pathlib.Path) -> None:
        """
        The cuBLAS shared library contains embedded cubins for many, but not all, architectures.
        """
        cubin = get_cubin(arch = parameters.arch, cublas = cublas, workdir = workdir)
        with ELF(file = cubin) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_EXEC'
            assert elf.is_cuda
            assert elf.arch == parameters.arch

    @pytest.fixture(scope = 'class')
    @staticmethod
    def version() -> semantic_version.Version:
        return nvcc.get_version()

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_embedded_cubin_cuinfo_and_tkinfo(self, version : semantic_version.Version, parameters : Parameters, cublas : CuBLAS, workdir : pathlib.Path) -> None:
        """
        Retrieve the `cuinfo` and `tkinfo` note sections from a cuBLAS cubin.
        """
        cubin = get_cubin(arch = parameters.arch, cublas = cublas, workdir = workdir)

        cuinfo, tkinfo = get_cuinfo_and_tkinfo(arch = parameters.arch, file = cubin)

        if cuinfo is not None:
            assert cuinfo.toolkit_version == int(f'{version.major}{version.minor}')

        if tkinfo is not None:
            assert tkinfo.tool_name == 'ptxas'
            assert f'-arch {parameters.arch.as_sm}' in tkinfo.tool_options
            assert '-m 64' in tkinfo.tool_options
            assert f'{version.major}.{version.minor}' in tkinfo.tool_version

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
class TestSaxpy:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELF` using an object file.
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

        * is a relocatable and not itself a cubin
        * contains an embedded cubin for the target architecture
        """
        with ELF(file = object_file) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_REL'
            assert not elf.is_cuda

        [name] = CuObjDump.extract_elf(file = object_file, arch = parameters.arch, name = 'saxpy', cwd = workdir)
        cubin = workdir / name
        assert cubin.is_file()

        with ELF(file = cubin) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_EXEC'
            assert elf.is_cuda
            assert elf.arch == parameters.arch

    def test_embedded_cubin_cuinfo_and_tkinfo(self, parameters : Parameters, object_file : pathlib.Path, workdir : pathlib.Path, cmake_file_api : cmake.FileAPI) -> None:
        """
        Retrieve the `cuinfo` and `tkinfo` note sections from a compiled output.
        """
        [name] = CuObjDump.extract_elf(file = object_file, arch = parameters.arch, name = 'saxpy', cwd = workdir)
        cubin = workdir / name
        assert cubin.is_file()

        cuinfo, tkinfo = get_cuinfo_and_tkinfo(arch = parameters.arch, file = cubin)

        if cuinfo is not None:
            assert cuinfo.virtual_sm == parameters.arch.compute_capability.as_int

        if tkinfo is not None:
            output            = subprocess.check_output(('nvcc', '--version')).decode()
            expt_tool_branch  = re.search(r'Build cuda_[0-9.r]+/compiler.[0-9_]+', output).group()
            expt_tool_version = re.search(r'Cuda compilation tools, release [0-9.]+, V[0-9.]+', output).group()
            expt_tool_options = ['-v', f'-arch {parameters.arch.as_sm}', '-m 64']

            match cmake_file_api.toolchains['CUDA']['compiler']['id']:
                case 'Clang':
                    expt_tool_options.append('-O 3')
                case _:
                    pass

            assert tkinfo.tool_branch  == expt_tool_branch
            assert tkinfo.tool_name    == 'ptxas'
            assert tkinfo.tool_version == expt_tool_version

            assert all(x in tkinfo.tool_options for x in expt_tool_options)
