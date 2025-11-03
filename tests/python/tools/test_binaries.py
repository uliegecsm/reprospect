import dataclasses
import functools
import json
import logging
import os
import pathlib
import shutil
import subprocess
import typing
import unittest

import pytest
import rich.console
import semantic_version

from reprospect.tools.architecture    import NVIDIAArch
from reprospect.tools.binaries        import CuObjDump, CuppFilt, LlvmCppFilt, ResourceUsage, Function
from reprospect.utils                 import cmake
from reprospect.utils.compile_command import get_arch_from_compile_command

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

@pytest.fixture(scope = 'session')
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

def test_get_arch_from_compile_command(cmake_file_api) -> None:
    """
    Test :py:meth:`reprospect.utils.compile_command.get_arch_from_compile_command`.
    """
    COMMANDS : dict[str, set] = {
        # For nvcc.
        '-arch=sm_86' : {NVIDIAArch.from_compute_capability(cc = 86)},
        '--gpu-architecture=compute_70 --gpu-code=sm_70' : {NVIDIAArch.from_str('VOLTA70')},
        '--generate-code=arch=compute_90,code=[sm_90]' : {NVIDIAArch.from_str('HOPPER90')},
        '--generate-code=arch=compute_90,code=[compute_90,sm_90]' : {NVIDIAArch.from_str('HOPPER90')},
        # For clang.
        '--cuda-gpu-arch=sm_120' : {NVIDIAArch.from_str('BLACKWELL120')},
        '--cuda-gpu-arch=sm_120 --cuda-gpu-arch=sm_86' : {NVIDIAArch.from_str('BLACKWELL120'), NVIDIAArch.from_str('AMPERE86')},
    }

    for command, arch in COMMANDS.items():
        assert get_arch_from_compile_command(cmd = command) == arch

    with open(pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'compile_commands.json', 'r', encoding = 'utf-8') as fin:
        compile_commands = json.load(fin)

    command = [x for x in compile_commands if x['file'].endswith('tests/cpp/cuda/test_saxpy.cpp')]
    assert len(command) == 1
    cmake_cuda_architecture = int(cmake_file_api.cache['CMAKE_CUDA_ARCHITECTURES']['value'].split('-')[0])
    assert get_arch_from_compile_command(cmd = command[0]['command']) == {NVIDIAArch.from_compute_capability(cc = cmake_cuda_architecture)}

def get_compilation_output(*,
    source : pathlib.Path,
    cwd : pathlib.Path,
    arch : NVIDIAArch,
    cmake_file_api : cmake.FileAPI,
    object : bool = True, # pylint: disable=redefined-builtin
    resource_usage : bool = False,
    ptx : bool = False,
) -> typing.Tuple[pathlib.Path, str]:
    """
    Compile the `source` in `cwd` for `arch`.
    """
    output = cwd / (source.stem + ('.o' if object else ''))

    cmd = [
        cmake_file_api.cache['CMAKE_CUDA_COMPILER_LAUNCHER']['value'],
        cmake_file_api.toolchains['CUDA']['compiler']['path'],
        '-std=c++20',
    ]

    # For compiling an executable, if the source ends with '.cpp', we need '-x cu' for nvcc and '-x cuda' for clang.
    if not object and source.suffix == '.cpp':
        match cmake_file_api.toolchains['CUDA']['compiler']['id']:
            case 'NVIDIA':
                cmd += ['-x', 'cu']
            case 'Clang':
                cmd += ['-x', 'cuda']
            case _:
                raise ValueError(f"unsupported compiler ID {cmake_file_api.toolchains['CUDA']['compiler']['id']}")

    # Clang tends to add a lot of debug code otherwise.
    cmd.append('-O3')

    match cmake_file_api.toolchains['CUDA']['compiler']['id']:
        case 'NVIDIA':
            cmd.append('--gpu-architecture=' + arch.as_compute)
            if ptx:
                cmd.append('--gpu-code=' + arch.as_compute + ',' + arch.as_sm)
            else:
                cmd.append('--gpu-code=' + arch.as_sm)
            if resource_usage:
                cmd.append('--resource-usage')
        case 'Clang':
            cmd.append(f'--cuda-gpu-arch={arch.as_sm}')
            if ptx:
                cmd.append('--cuda-include-ptx=' + arch.as_sm)
            if resource_usage:
                cmd += ['-Xcuda-ptxas', '-v',]
        case _:
            raise ValueError(f"unsupported compiler ID {cmake_file_api.toolchains['CUDA']['compiler']['id']}")

    cmd += [
        '-c', source,
        '-o', output,
    ]

    logging.info(f'Compiling {source} with {cmd} in {cwd}.')

    return (output, subprocess.check_output(
        args = cmd,
        cwd = cwd,
        stderr = subprocess.STDOUT,
    ).decode())

@dataclasses.dataclass(frozen = True)
class Parameters:
    arch : NVIDIAArch

@functools.cache
def architectures(version : semantic_version = semantic_version.Version(os.environ['CUDA_VERSION'])) -> list[NVIDIAArch]:
    """
    Get the list of architectures to test, that are supported by `version`.
    """
    return tuple({
        arch for cc in [
            70,
            75,
            80,
            86,
            89,
            90,
            100,
            120,
        ] if (arch := NVIDIAArch.from_compute_capability(cc = cc)).compute_capability.supported(version = version)
    })

PARAMETERS = [Parameters(arch = arch) for arch in architectures()]

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestResourceUsage:
    """
    All tests related to reading resource usage from compilation output.
    """
    class TestSharedMemory:
        """
        When the kernel uses shared memory.
        """
        FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'shared_memory.cu'

        def test(self, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check how shared memory is reported in the compilation output.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = TMPDIR,
                arch = parameters.arch,
                object = True,
                resource_usage = True,
                cmake_file_api = cmake_file_api,
            )

            assert f"Compiling entry function '_Z20shared_memory_kernelPfj' for '{parameters.arch.as_sm}'" in compilation, compilation

            if parameters.arch.compute_capability < 90:
                assert f'Used 10 registers, used 1 barriers, {128 * 4} bytes smem, 364 bytes cmem[0]' in compilation, compilation
            else:
                assert f'Used 10 registers, used 1 barriers, {128 * 4} bytes smem' in compilation, compilation

    class TestWideLoadStore:
        """
        When the kernel uses wide loads and stores.
        """
        FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'wide_load_store.cu'

        def test(self, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check how wide loads and stores influence the compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = TMPDIR,
                arch = parameters.arch,
                object = True,
                resource_usage = True,
                cmake_file_api = cmake_file_api,
            )

            assert f"Compiling entry function '_Z22wide_load_store_kernelP15MyAlignedStructIdEPKS0_' for '{parameters.arch.as_sm}'" in compilation, compilation

            if parameters.arch.compute_capability < 90:
                assert 'Used 12 registers, used 0 barriers, 368 bytes cmem[0]' in compilation, compilation
            elif parameters.arch.compute_capability < 100:
                expt_regs = {
                    'NVIDIA' : 14,
                    'Clang' : 12,
                }[cmake_file_api.toolchains['CUDA']['compiler']['id']]
                assert f'Used {expt_regs} registers, used 0 barriers' in compilation, compilation
            else:
                assert 'Used 12 registers, used 0 barriers' in compilation, compilation

    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.
        """
        FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'

        def test(self, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = TMPDIR,
                arch = parameters.arch,
                object = True,
                resource_usage = True,
                cmake_file_api = cmake_file_api,
            )

            assert f"Compiling entry function '_Z12saxpy_kernelfPKfPfj' for '{parameters.arch.as_sm}'" in compilation, compilation

            if parameters.arch.compute_capability < 90:
                assert 'Used 10 registers, used 0 barriers, 380 bytes cmem[0]' in compilation, compilation
            else:
                assert 'Used 10 registers, used 0 barriers' in compilation, compilation

class TestCuppFilt:
    """
    Test :py:class:`reprospect.tools.binaries.CuppFilt`.
    """
    def test_demangle(self):
        assert CuppFilt.demangle(s = '_Z5saxpyfPKfPfj') == 'saxpy(float, const float *, float *, unsigned int)'

@pytest.mark.skipif(
    shutil.which(LlvmCppFilt.get_executable()) is None, reason = f'requires that {LlvmCppFilt.get_executable()} is installed'
)
class TestLlvmCppFilt:
    """
    Test :py:class:`reprospect.tools.binaries.LlvmCppFilt`.
    """
    def test_demangle(self):
        MANGLED = '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj'
        assert LlvmCppFilt.demangle(s = MANGLED) == 'void add_and_increment_kernel<0u>(unsigned int*)'
        # cu++filt cannot demangle this symbol.
        assert CuppFilt.demangle(s = MANGLED).startswith('_Z')

class TestFunction:
    """
    Tests related to :py:class:`reprospect.tools.binaries.Function`.
    """
    CODE = """\
        .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
        /*0000*/                   LDC R1, c[0x0][0x37c]                &wr=0x0          ?trans1;           /* 0x0000df00ff017b82 */
                                                                                                            /* 0x000e220000000800 */
        /*0010*/                   S2R R0, SR_TID.X                     &wr=0x1          ?trans7;           /* 0x0000000000007919 */
                                                                                                            /* 0x000e6e0000002100 */
"""

    RU = {
        ResourceUsage.REGISTER : 10,
        ResourceUsage.STACK    : 0,
        ResourceUsage.SHARED   : 0,
        ResourceUsage.LOCAL    : 0,
        ResourceUsage.CONSTANT : {0: 924},
        ResourceUsage.TEXTURE  : 0,
        ResourceUsage.SURFACE  : 0,
        ResourceUsage.SAMPLER  : 0
    }

    def test_string_representation(self) -> None:
        """
        Test :py:meth:`reprospect.tools.binaries.Function.__str__`.
        """
        function = Function(code = self.CODE, ru = self.RU)

        # Check that the conversion to a table with truncation of long lines works as expected.
        with rich.console.Console(width = 200) as console, console.capture() as capture:
            console.print(function.to_table(max_code_length = 120), no_wrap = True)

        assert capture.get() == """\
┌─────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Code            │ .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"                                                       │
│                 │ /*0000*/                   LDC R1, c[0x0][0x37c]                &wr=0x0          ?trans1;           /* 0x0000df00ff017b… │
│                 │                                                                                                     /* 0x000e2200000008… │
│                 │ /*0010*/                   S2R R0, SR_TID.X                     &wr=0x1          ?trans7;           /* 0x00000000000079… │
│                 │                                                                                                     /* 0x000e6e00000021… │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Resource usage  │ REG: 10, STACK: 0, SHARED: 0, LOCAL: 0, CONSTANT: {0: 924}, TEXTURE: 0, SURFACE: 0, SAMPLER: 0                           │
└─────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

        # Check the rich representation with the default table size.
        assert str(function) == """\
┌─────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Code            │ .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"                                                                 │
│                 │ /*0000*/                   LDC R1, c[0x0][0x37c]                &wr=0x0          ?trans1;           /* 0x0000df00ff017b82 */       │
│                 │                                                                                                     /* 0x000e220000000800 */       │
│                 │ /*0010*/                   S2R R0, SR_TID.X                     &wr=0x1          ?trans7;           /* 0x0000000000007919 */       │
│                 │                                                                                                     /* 0x000e6e0000002100 */       │
├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Resource usage  │ REG: 10, STACK: 0, SHARED: 0, LOCAL: 0, CONSTANT: {0: 924}, TEXTURE: 0, SURFACE: 0, SAMPLER: 0                                     │
└─────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

class TestCuObjDump:
    """
    Tests related to :py:class:`reprospect.tools.binaries.CuObjDump`.
    """
    @pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.
        """
        CPP_FILE  = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cpp'
        CUDA_FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'
        SYMBOL    = '_Z12saxpy_kernelfPKfPfj'
        SIGNATURE = CuppFilt.demangle(SYMBOL)

        def test_sass_from_object(self, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CUDA_FILE` as object, extract SASS and analyse resource usage.
            """
            output, _ = get_compilation_output(
                source = self.CUDA_FILE,
                cwd = TMPDIR,
                arch = parameters.arch,
                object = True,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump = CuObjDump(file = output, arch = parameters.arch, sass = True)

            sass = output.with_suffix(f'.{parameters.arch.compute_capability}.sass')
            logging.debug(f'Writing SASS to {sass}.')
            sass.write_text(cuobjdump.sass)

            assert len(cuobjdump.functions) == 1

            if parameters.arch.compute_capability < 90:
                cst = 380
            elif parameters.arch.compute_capability < 100:
                cst = 556
            else:
                cst = 924

            assert cuobjdump.functions[self.SIGNATURE].ru == {
                ResourceUsage.REGISTER: 10,
                ResourceUsage.STACK: 0,
                ResourceUsage.SHARED: 0,
                ResourceUsage.LOCAL: 0,
                ResourceUsage.CONSTANT: {0: cst},
                ResourceUsage.TEXTURE: 0,
                ResourceUsage.SURFACE: 0,
                ResourceUsage.SAMPLER: 0,
            }, cuobjdump.functions[self.SIGNATURE].ru

        def test_extract_cubin_from_file(self, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the `cubin` from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = TMPDIR,
                arch = parameters.arch,
                object = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = TMPDIR,
                cubin = output.name,
            )

            assert cubin.is_file()

            assert len(cuobjdump.functions) == 1
            assert self.SIGNATURE in cuobjdump.functions

        def test_extract_symbol_table(self, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the symbol table from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = TMPDIR,
                arch = parameters.arch,
                object = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = TMPDIR,
                cubin = output.name,
                sass = False,
            )

            symbols = cuobjdump.symtab(cubin = cubin, arch = parameters.arch)

            assert self.SYMBOL in symbols['name'].values

    def test_string_representation(self) -> None:
        """
        Test :py:meth:`reprospect.tools.binaries.CuObjDump.__str__`.
        """
        def mock_init(self):
            self.file = pathlib.Path('code_object.o')
            self.arch = NVIDIAArch.from_str('BLACKWELL120')
            self.functions = {
                'my_kernel(float, const float *, float *, unsigned int)' : Function(
                    code = TestFunction.CODE,
                    ru = TestFunction.RU
                ),
                'my_other_kernel(float, const float *, float *, unsigned int)' : Function(
                    code = TestFunction.CODE,
                    ru = TestFunction.RU
                )
            }

        with unittest.mock.patch.object(CuObjDump, "__init__", mock_init):
            cuobjdump = CuObjDump() # pylint: disable=no-value-for-parameter
            assert str(cuobjdump) == """\
CuObjDump of code_object.o for architecture BLACKWELL120:
┌─────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Function        │ my_kernel(float, const float *, float *, unsigned int)                                                                             │
├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Code            │ .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"                                                                 │
│                 │ /*0000*/                   LDC R1, c[0x0][0x37c]                &wr=0x0          ?trans1;           /* 0x0000df00ff017b82 */       │
│                 │                                                                                                     /* 0x000e220000000800 */       │
│                 │ /*0010*/                   S2R R0, SR_TID.X                     &wr=0x1          ?trans7;           /* 0x0000000000007919 */       │
│                 │                                                                                                     /* 0x000e6e0000002100 */       │
├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Resource usage  │ REG: 10, STACK: 0, SHARED: 0, LOCAL: 0, CONSTANT: {0: 924}, TEXTURE: 0, SURFACE: 0, SAMPLER: 0                                     │
└─────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Function        │ my_other_kernel(float, const float *, float *, unsigned int)                                                                       │
├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Code            │ .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"                                                                 │
│                 │ /*0000*/                   LDC R1, c[0x0][0x37c]                &wr=0x0          ?trans1;           /* 0x0000df00ff017b82 */       │
│                 │                                                                                                     /* 0x000e220000000800 */       │
│                 │ /*0010*/                   S2R R0, SR_TID.X                     &wr=0x1          ?trans7;           /* 0x0000000000007919 */       │
│                 │                                                                                                     /* 0x000e6e0000002100 */       │
├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Resource usage  │ REG: 10, STACK: 0, SHARED: 0, LOCAL: 0, CONSTANT: {0: 924}, TEXTURE: 0, SURFACE: 0, SAMPLER: 0                                     │
└─────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
