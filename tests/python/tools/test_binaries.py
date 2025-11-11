import logging
import os
import pathlib
import shutil
import typing

import pytest
import rich.console

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump, CuppFilt, LlvmCppFilt, ResourceUsage, Function
from reprospect.tools.binaries     import cuobjdump
from reprospect.utils              import cmake

from tests.python.compilation import get_compilation_output, get_cubin_name
from tests.python.parameters  import Parameters, PARAMETERS

@pytest.fixture(scope = 'session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

@pytest.fixture(scope = 'session')
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestResourceUsage:
    """
    All tests related to reading resource usage from compilation output.
    """
    class TestSharedMemory:
        """
        When the kernel uses shared memory.
        """
        FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'test_binaries' / 'shared_memory.cu'

        def test(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check how shared memory is reported in the compilation output.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = workdir,
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
        FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'test_binaries' / 'wide_load_store.cu'

        def test(self, workdir : pathlib.Path, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check how wide loads and stores influence the compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = workdir,
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
        FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'

        def test(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = workdir,
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
    CODE : typing.Final[str] = """\
        .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
        /*0000*/                   LDC R1, c[0x0][0x37c]                &wr=0x0          ?trans1;           /* 0x0000df00ff017b82 */
                                                                                                            /* 0x000e220000000800 */
        /*0010*/                   S2R R0, SR_TID.X                     &wr=0x1          ?trans7;           /* 0x0000000000007919 */
                                                                                                            /* 0x000e6e0000002100 */
"""

    RU : typing.Final[cuobjdump.ResourceUsageDict] = {
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
        CPP_FILE  : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cpp'
        CUDA_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'
        SYMBOL    : typing.Final[str] = '_Z12saxpy_kernelfPKfPfj'
        SIGNATURE : typing.Final[str] = CuppFilt.demangle(SYMBOL)

        def test_sass_from_object(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CUDA_FILE` as object, extract SASS and analyse resource usage.
            """
            output, _ = get_compilation_output(
                source = self.CUDA_FILE,
                cwd = workdir,
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

        def test_extract_cubin_from_file(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the `cubin` from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                ),
            )

            assert cubin.is_file()

            assert len(cuobjdump.functions) == 1
            assert self.SIGNATURE in cuobjdump.functions

        def test_extract_symbol_table(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the symbol table from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                ),
                sass = False,
            )

            symbols = cuobjdump.symtab(cubin = cubin, arch = parameters.arch)

            assert self.SYMBOL in symbols['name'].values

    @pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
    class TestMany:
        """
        When there are many kernels.

        .. note::

            ``__device__`` functions have been inlined.
        """
        CUDA_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'test_binaries' / 'many.cu'
        CPP_FILE  : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'test_binaries' / 'many.cpp'

        SIGNATURES : typing.Final[dict[str, str]] = {
            '_Z6say_hiv' : 'say_hi()',
            '_Z20vector_atomic_add_42PKfS0_Pfj' : 'vector_atomic_add_42(const float *, const float *, float *, unsigned int)',
        }

        def test_sass_from_object(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CUDA_FILE` as object, extract SASS.
            """
            output, _ = get_compilation_output(
                source = self.CUDA_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object = True,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump = CuObjDump(file = output, arch = parameters.arch, sass = True)

            sass = output.with_suffix(f'.{parameters.arch.compute_capability}.sass')
            logging.debug(f'Writing SASS to {sass}.')
            sass.write_text(cuobjdump.sass)

            assert len(cuobjdump.functions) == len(self.SIGNATURES)

        def test_extract_cubin_from_file(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the `cubin` from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                ),
            )

            assert cubin.is_file()

            assert set(cuobjdump.functions.keys()) == set(self.SIGNATURES.values())

        def test_extract_symbol_table(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the symbol table from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                ),
                sass = False,
            )

            symbols = cuobjdump.symtab(cubin = cubin, arch = parameters.arch)

            assert all(x in symbols['name'].values for x in self.SIGNATURES), symbols

    def test_string_representation(self) -> None:
        """
        Test :py:meth:`reprospect.tools.binaries.CuObjDump.__str__`.
        """
        cuobjdump = CuObjDump(
            file = pathlib.Path('code_object.o'),
            arch = NVIDIAArch.from_str('BLACKWELL120'),
            sass = False,
        )
        cuobjdump.functions = {
            'my_kernel(float, const float *, float *, unsigned int)' : Function(
                code = TestFunction.CODE,
                ru = TestFunction.RU
            ),
            'my_other_kernel(float, const float *, float *, unsigned int)' : Function(
                code = TestFunction.CODE,
                ru = TestFunction.RU
            )
        }

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
