import functools
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
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.binaries.elf import (
    ELF,
    CuInfo,
    NvInfo,
    NvInfoEIATTR,
    NvInfoEIFMT,
    NvInfoEntry,
    TkInfo,
)
from reprospect.utils import cmake, nvcc

from tests.compilation import get_compilation_output, get_cubin_name
from tests.cublas import CuBLAS
from tests.parameters import PARAMETERS, Parameters


@functools.cache
def nvcc_version() -> semantic_version.Version:
    return nvcc.get_version()

def check_version(version: int | str) -> None:
    """
    Check `version` against ``nvcc`` version.

    .. note::

        As of CUDA 13.1, tools report 13.0.
    """
    ver = nvcc_version()

    expt_minor = 0 if ver in semantic_version.SimpleSpec('==13.1') else ver.minor

    if isinstance(version, int):
        assert version == int(f'{ver.major}{expt_minor}')
    else:
        assert f'{ver.major}.{expt_minor}' in version

class TestGetComputeCapabilityFromEFlags:
    CC_E_FLAGS: typing.Final[dict[int, int]] = {
        70: 0b10001100000010101000110,      # 0x460546
        75: 0b10010110000010101001011,      # 0x4b054b
        80: 0b10100000000010101010000,      # 0x500550
        86: 0b10101100000010101010110,      # 0x560556
        89: 0b10110010000010101011001,      # 0x590559
        90: 0b10110100000010101011010,      # 0x5a055a
        100: 0b110000000000110010000000010, # 0x6006402
        103: 0b110000000000110011100000010, # 0x6006702
        110: 0b110000000000110111000000010, # 0x6006e02
        120: 0b110000000000111100000000010, # 0x6007802
        121: 0b110000000000111100100000010, # 0x6007902
    }
    """
    Values of `e_flags` obtained by calling ``cuobjdump --dump-elf`` on a cubin
    and reading the value of `flags` from the header for each architecture.
    """

    @pytest.mark.parametrize(('cc', 'e_flags'), CC_E_FLAGS.items())
    def test(self, cc: int, e_flags: int) -> None:
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

    @pytest.mark.parametrize(('cc', 'e_flags'), CC_E_FLAGS.items())
    def test_get_compute_capability_from_e_flags(self, cc: int, e_flags: int) -> None:
        assert ELF.compute_capability(e_flags).as_int == cc

class TestCUDART:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELF` using the CUDA runtime shared library.
    """
    @pytest.fixture(scope='class')
    def cudart(self, cmake_file_api: cmake.FileAPI) -> pathlib.Path:
        cudart = pathlib.Path(cmake_file_api.cache['CUDA_CUDART']['value'])
        assert cudart.is_file()
        return cudart

    def test_shared_library(self, cudart: pathlib.Path) -> None:
        """
        The CUDA runtime shared library is a shared library and not itself a cubin.
        """
        with ELF(file=cudart) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_DYN'
            assert not elf.is_cuda

    def test_embedded_cubin(self, cudart: pathlib.Path) -> None:
        """
        The CUDA runtime shared library does not contain embedded cubins.
        """
        with pytest.raises(subprocess.CalledProcessError) as exc:
            tuple(CuObjDump.list_elf(file=cudart))
        assert 'does not contain device code' in exc.value.stderr

def get_cubin(arch: NVIDIAArch, cublas: CuBLAS, workdir: pathlib.Path) -> pathlib.Path:
    try:
        [cubin] = cublas.extract(arch=arch, cwd=workdir, randomly=True)
    except IndexError:
        pytest.skip(f'{cublas} does not contain an embedded cubin for {arch}.')
    assert cubin.is_file()
    return cubin

def get_cuinfo_and_tkinfo(*, arch: NVIDIAArch, file: pathlib.Path, version: semantic_version.Version = semantic_version.Version(os.environ['CUDA_VERSION'])) -> tuple[CuInfo | None, TkInfo | None]: # noqa: B008
    """
    Extract `cuinfo` and `tkinfo` note sections.

    It takes care of checking if each note section has to exist or not, given the `arch` and CUDA `version`.
    """
    with ELF(file=file) as elf:

        def has(*, name: str) -> bool:
            return elf.elf.has_section(section_name='.note.nv.' + name)

        def get(*, name: str) -> elftools.elf.sections.NoteSection:
            section = elf.elf.get_section_by_name(name='.note.nv.' + name)
            assert isinstance(section, elftools.elf.sections.NoteSection)
            return section

        cuinfos: tuple[CuInfo, ...] | None = tuple(note for note in CuInfo.decode(note=get(name='cuinfo'))) if has(name='cuinfo') else None
        tkinfos: tuple[TkInfo, ...] | None = tuple(note for note in TkInfo.decode(note=get(name='tkinfo'))) if has(name='tkinfo') else None

        cuinfo: CuInfo | None = None
        if version in semantic_version.SimpleSpec('<13.0.0'):
            assert cuinfos is None
        else:
            logging.info(cuinfos)
            assert cuinfos is not None
            assert len(cuinfos) == 1
            cuinfo = cuinfos[0]
            assert cuinfo.virtual_sm == arch.compute_capability.as_int

        tkinfo: TkInfo | None = None
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
    @pytest.fixture(scope='class')
    def cublas(self, cmake_file_api: cmake.FileAPI) -> CuBLAS:
        return CuBLAS(cmake_file_api=cmake_file_api)

    def test_shared_library(self, cublas: CuBLAS) -> None:
        """
        The cuBLAS shared library is a shared library and not itself a cubin.
        """
        with ELF(file=cublas.libcublas) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_DYN'
            assert not elf.is_cuda

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test_embedded_cubin(self, parameters: Parameters, cublas: CuBLAS, workdir: pathlib.Path) -> None:
        """
        The cuBLAS shared library contains embedded cubins for many, but not all, architectures.
        """
        cubin = get_cubin(arch=parameters.arch, cublas=cublas, workdir=workdir)
        with ELF(file=cubin) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_EXEC'
            assert elf.is_cuda
            assert elf.arch == parameters.arch

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test_embedded_cubin_cuinfo_and_tkinfo(self, parameters: Parameters, cublas: CuBLAS, workdir: pathlib.Path) -> None:
        """
        Retrieve the `cuinfo` and `tkinfo` note sections from a cuBLAS cubin.
        """
        cubin = get_cubin(arch=parameters.arch, cublas=cublas, workdir=workdir)

        cuinfo, tkinfo = get_cuinfo_and_tkinfo(arch=parameters.arch, file=cubin)

        if cuinfo is not None:
            check_version(version=cuinfo.toolkit_version)

        if tkinfo is not None:
            assert tkinfo.tool_name == 'ptxas'
            assert f'-arch {parameters.arch.as_sm}' in tkinfo.tool_options
            assert '-m 64' in tkinfo.tool_options
            check_version(version=tkinfo.tool_version)

@pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
class TestSaxpy:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.ELF` using an object file.
    """
    FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cu'

    @pytest.fixture
    def object_file(self, parameters: Parameters, cmake_file_api: cmake.FileAPI, workdir: pathlib.Path) -> pathlib.Path:
        """
        Compile into object file.
        """
        object_file, _ = get_compilation_output(
            source=self.FILE,
            cwd=workdir,
            arch=parameters.arch,
            object_file=True,
            resource_usage=True,
            cmake_file_api=cmake_file_api,
        )
        assert object_file.is_file()
        return object_file

    def test_object_file_and_embedded_cubin(self, parameters: Parameters, object_file: pathlib.Path, workdir: pathlib.Path) -> None:
        """
        Check that the object file:

        * is a relocatable and not itself a cubin
        * contains an embedded cubin for the target architecture
        """
        with ELF(file=object_file) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_REL'
            assert not elf.is_cuda

        [name] = CuObjDump.extract_elf(file=object_file, arch=parameters.arch, name='saxpy', cwd=workdir)
        cubin = workdir / name
        assert cubin.is_file()

        with ELF(file=cubin) as elf:
            logging.info(elf.header)

            assert elf.header['e_type'] == 'ET_EXEC'
            assert elf.is_cuda
            assert elf.arch == parameters.arch

    def test_embedded_cubin_cuinfo_and_tkinfo(self, parameters: Parameters, object_file: pathlib.Path, workdir: pathlib.Path, cmake_file_api: cmake.FileAPI) -> None:
        """
        Retrieve the `cuinfo` and `tkinfo` note sections from a compiled output.
        """
        [name] = CuObjDump.extract_elf(file=object_file, arch=parameters.arch, name='saxpy', cwd=workdir)
        cubin = workdir / name
        assert cubin.is_file()

        cuinfo, tkinfo = get_cuinfo_and_tkinfo(arch=parameters.arch, file=cubin)

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

class TestNvInfo:
    """
    Tests for :py:class:`reprospect.tools.binaries.elf.NvInfo`.
    """
    DATA_0: typing.Final[bytes] = b'\x047\x04\x00\x82\x00\x00\x00\x015\x00\x00\x04\n\x08\x00\t\x00\x00\x00`\x018\x00\x03\x198\x00\x04\x17\x0c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\xe1\x00\x03\x1b\xff\x00\x03_\x00\x00\x04\x1c\x08\x00`\x00\x00\x00p\x01\x00\x00\x04\x05\x0c\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x04\x1e\x04\x00\x00\x00\x00\x00'
    DATA_1: typing.Final[bytes] = b'\x047\x04\x00\x82\x00\x00\x00\x015\x00\x00\x04\n\x08\x00\x0c\x00\x00\x00`\x01x\x00\x03\x19x\x00\x04\x17\x0c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\xe1\x01\x03\x1b\xff\x00\x03_\x00\x00\x04\x1c\x04\x00\x10\x00\x00\x00'
    DATA_2: typing.Final[bytes] = b'\x047\x04\x00\x82\x00\x00\x00\x015\x00\x00\x04\n\x08\x00\x0f\x00\x00\x00`\x01x\x00\x03\x19x\x00\x04\x17\x0c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\xe1\x01\x03\x1b\xff\x00\x03_\x00\x00\x04\x1c\x04\x00\x10\x00\x00\x00'
    DATA_3: typing.Final[bytes] = b'\x046\x04\x00\x01\x00\x00\x00\x047\x04\x00\x80\x00\x00\x00\x04\n\x08\x00\x02\x00\x00\x00`\x01\x1c\x00\x03\x19\x1c\x00\x04\x17\x0c\x00\x00\x00\x00\x00\x03\x00\x18\x00\x00\xf0\x11\x00\x04\x17\x0c\x00\x00\x00\x00\x00\x02\x00\x10\x00\x00\xf0!\x00\x04\x17\x0c\x00\x00\x00\x00\x00\x01\x00\x08\x00\x00\xf0!\x00\x04\x17\x0c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x11\x00\x03\x1b\xff\x00\x041\x04\x00\x10\x00\x00\x00\x04\x1c\x08\x00`\x00\x00\x00\xe0\x00\x00\x00'

    def test_parse_data_0(self) -> None:
        """
        Parse :py:attr:`DATA_0`.
        """
        assert NvInfo.parse(data=self.DATA_0).attributes == (
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.CUDA_API_VERSION,    value=(int('0x82', base=16),)),
            NvInfoEntry(eifmt=NvInfoEIFMT.NVAL, eiattr=NvInfoEIATTR.SW2861232_WAR,       value=None),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.PARAM_CBANK,         value=(int('0x9', base=16), int('0x380160', base=16))),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.CBANK_PARAM_SIZE,    value=int('0x38', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.KPARAM_INFO,         value=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\xe1\x00'),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MAXREG_COUNT,        value=int('0xff', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MERCURY_ISA_VERSION, value=0),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.EXIT_INSTR_OFFSETS,  value=(int('0x60', base=16), int('0x170', base=16))),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.MAX_THREADS,         value=3 * (int('0x1', base=16),)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.CRS_STACK_SIZE,      value=(int('0x0', base=16),)),
        )

    def test_parse_data_1(self) -> None:
        """
        Parse :py:attr:`DATA_1`.
        """
        assert NvInfo.parse(data=self.DATA_1).attributes == (
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.CUDA_API_VERSION,    value=(int('0x82', base=16),)),
            NvInfoEntry(eifmt=NvInfoEIFMT.NVAL, eiattr=NvInfoEIATTR.SW2861232_WAR,       value=None),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.PARAM_CBANK,         value=(int('0xc', base=16), int('0x780160', base=16))),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.CBANK_PARAM_SIZE,    value=int('0x78', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.KPARAM_INFO,         value=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\xe1\x01'),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MAXREG_COUNT,        value=int('0xff', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MERCURY_ISA_VERSION, value=0),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.EXIT_INSTR_OFFSETS,  value=(int('0x10', base=16),)),
        )

    def test_parse_data_2(self) -> None:
        """
        Parse :py:attr:`DATA_2`.
        """
        assert NvInfo.parse(data=self.DATA_2).attributes == (
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.CUDA_API_VERSION,    value=(int('0x82', base=16),)),
            NvInfoEntry(eifmt=NvInfoEIFMT.NVAL, eiattr=NvInfoEIATTR.SW2861232_WAR,       value=None),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.PARAM_CBANK,         value=(int('0xf', base=16), int('0x780160', base=16))),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.CBANK_PARAM_SIZE,    value=int('0x78', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.KPARAM_INFO,         value=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\xe1\x01'),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MAXREG_COUNT,        value=int('0xff', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MERCURY_ISA_VERSION, value=0),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.EXIT_INSTR_OFFSETS,  value=(int('0x10', base=16),)),
        )

    def test_parse_data_3(self) -> None:
        """
        Parse :py:attr:`DATA_3`.

        ``cuobjdump`` parses :py:attr:`DATA_3` as::

            <0x1>
            Attribute: EIATTR_SW_WAR
            Format: EIFMT_SVAL
            Value: 0x1
            <0x2>
            Attribute: EIATTR_CUDA_API_VERSION
            Format: EIFMT_SVAL
            Value: 0x80
            <0x3>
            Attribute: EIATTR_PARAM_CBANK
            Format: EIFMT_SVAL
            Value: 0x2 0x1c0160
            <0x4>
            Attribute: EIATTR_CBANK_PARAM_SIZE
            Format: EIFMT_HVAL
            Value: 0x1c
            <0x5>
            Attribute: EIATTR_KPARAM_INFO
            Format: EIFMT_SVAL
            Value: Index : 0x0 Ordinal : 0x3 Offset  : 0x18 Size    : 0x4
                Pointee's logAlignment : 0x0 Space : 0x0 cbank : 0x1f Parameter Space : CBANK
            <0x6>
            Attribute: EIATTR_KPARAM_INFO
            Format: EIFMT_SVAL
            Value: Index : 0x0 Ordinal : 0x2 Offset  : 0x10 Size    : 0x8
                Pointee's logAlignment : 0x0 Space : 0x0 cbank : 0x1f Parameter Space : CBANK
            <0x7>
            Attribute: EIATTR_KPARAM_INFO
            Format: EIFMT_SVAL
            Value: Index : 0x0 Ordinal : 0x1 Offset  : 0x8 Size    : 0x8
                Pointee's logAlignment : 0x0 Space : 0x0 cbank : 0x1f Parameter Space : CBANK
            <0x8>
            Attribute: EIATTR_KPARAM_INFO
            Format: EIFMT_SVAL
            Value: Index : 0x0 Ordinal : 0x0 Offset  : 0x0 Size    : 0x4
                Pointee's logAlignment : 0x0 Space : 0x0 cbank : 0x1f Parameter Space : CBANK
            <0x9>
            Attribute: EIATTR_MAXREG_COUNT
            Format: EIFMT_HVAL
            Value: 0xff
            <0x10>
            Attribute: EIATTR_INT_WARP_WIDE_INSTR_OFFSETS
            Format: EIFMT_SVAL
            Value: 0x10
            <0x11>
            Attribute: EIATTR_EXIT_INSTR_OFFSETS
            Format: EIFMT_SVAL
            Value: 0x60 0xe0
        """
        assert NvInfo.parse(data=self.DATA_3).attributes == (
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.SW_WAR,                  value=(int('0x1', base=16),)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.CUDA_API_VERSION,        value=(int('0x80', base=16),)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.PARAM_CBANK,             value=(int('0x2', base=16), int('0x1c0160', base=16))),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.CBANK_PARAM_SIZE,        value=int('0x1c', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.KPARAM_INFO,             value=b'\x00\x00\x00\x00\x03\x00\x18\x00\x00\xf0\x11\x00'),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.KPARAM_INFO,             value=b'\x00\x00\x00\x00\x02\x00\x10\x00\x00\xf0!\x00'),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.KPARAM_INFO,             value=b'\x00\x00\x00\x00\x01\x00\x08\x00\x00\xf0!\x00'),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.KPARAM_INFO,             value=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x11\x00'),
            NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MAXREG_COUNT,            value=int('0xff', base=16)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.WARP_WIDE_INSTR_OFFSETS, value=(int('0x10', base=16),)),
            NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.EXIT_INSTR_OFFSETS,      value=(int('0x60', base=16), int('0xe0', base=16))),
        )

    @pytest.fixture(scope='class')
    @staticmethod
    def version() -> semantic_version.Version:
        return nvcc.get_version()

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test(self, version, parameters: Parameters, cmake_file_api: cmake.FileAPI, workdir: pathlib.Path) -> None:
        """
        Extract the `.nv.info.<mangled>` section of the kernel.
        """
        FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cpp'

        output, _ = get_compilation_output(
            source=FILE,
            cwd=workdir,
            arch=parameters.arch,
            object_file=False,
            resource_usage=False,
            cmake_file_api=cmake_file_api,
        )
        _, cubin = CuObjDump.extract(
            file=output,
            arch=parameters.arch,
            cwd=workdir,
            sass=False,
            cubin=get_cubin_name(
                compiler_id=cmake_file_api.toolchains['CUDA']['compiler']['id'],
                file=output,
                arch=parameters.arch,
                object_file=False,
            ),
        )
        with ELF(file=cubin) as elf:
            assert {
                NvInfoEntry(eifmt=NvInfoEIFMT.SVAL, eiattr=NvInfoEIATTR.CUDA_API_VERSION, value=(int(f'{version.major}{version.minor}'),)),
                NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.CBANK_PARAM_SIZE, value=int('0x1c', base=16)),
                NvInfoEntry(eifmt=NvInfoEIFMT.HVAL, eiattr=NvInfoEIATTR.MAXREG_COUNT,     value=int('0xff', base=16)),
            }.issubset(elf.nvinfo(mangled='_Z12saxpy_kernelfPKfPfj').attributes)
