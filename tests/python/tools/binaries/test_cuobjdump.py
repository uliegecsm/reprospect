import logging
import pathlib
import random
import typing

import pytest

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump, CuppFilt, Function, ResourceType, ResourceUsage
from reprospect.utils              import cmake
from reprospect.utils              import rich_helpers

from tests.python.compilation import get_compilation_output, get_cubin_name
from tests.python.cublas      import CuBLAS
from tests.python.parameters  import Parameters, PARAMETERS

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

    RU : typing.Final[ResourceUsage] = ResourceUsage({
        ResourceType.REGISTER : 10,
        ResourceType.STACK    : 0,
        ResourceType.SHARED   : 0,
        ResourceType.LOCAL    : 0,
        ResourceType.CONSTANT : {0: 924},
        ResourceType.TEXTURE  : 0,
        ResourceType.SURFACE  : 0,
        ResourceType.SAMPLER  : 0
    })

    def test_string_representation(self) -> None:
        """
        Test :py:meth:`reprospect.tools.binaries.Function.__str__`.
        """
        function = Function(code = self.CODE, ru = self.RU)

        # Check that the conversion to a table with truncation of long lines works as expected.
        rt = function.to_table(max_code_length = 120)
        assert rich_helpers.to_string(rt) == """\
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
    @staticmethod
    def dump(*, file : pathlib.Path, cuobjdump : CuObjDump) -> None:
        logging.info(f'Writing parsed SASS to {file}.')
        file.write_text(str(cuobjdump))

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.
        """
        CPP_FILE  : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cpp'
        CUDA_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cu'
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
                object_file = True,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump = CuObjDump(file = output, arch = parameters.arch, sass = True)

            assert not cuobjdump.file_is_cubin

            TestCuObjDump.dump(
                file = output.with_suffix(f'.{parameters.arch.compute_capability}.sass'),
                cuobjdump = cuobjdump,
            )

            assert len(cuobjdump.functions) == 1

            if parameters.arch.compute_capability < 90:
                cst = 380
            elif parameters.arch.compute_capability < 100:
                cst = 556
            else:
                cst = 924

            assert cuobjdump.functions[self.SIGNATURE].ru == ResourceUsage({
                ResourceType.REGISTER: 10,
                ResourceType.STACK: 0,
                ResourceType.SHARED: 0,
                ResourceType.LOCAL: 0,
                ResourceType.CONSTANT: {0: cst},
                ResourceType.TEXTURE: 0,
                ResourceType.SURFACE: 0,
                ResourceType.SAMPLER: 0,
            }), cuobjdump.functions[self.SIGNATURE].ru

        def test_extract_cubin_from_file(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the `cubin` from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = False,
                cmake_file_api = cmake_file_api,
            )

            # Extract the embedded CUDA binary file and check that it contains the expected function.
            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                    object_file = False,
                ),
            )

            assert cubin.is_file()
            assert cuobjdump.file_is_cubin

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
                object_file = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump = CuObjDump(file = output, arch = parameters.arch, sass = False)

            # This binary is not compiled to an object file. With `nvcc` as the compiler, the binary then contains
            # more than one embedded CUDA binary file. Hence, check that calling symtab raises.
            if cmake_file_api.toolchains['CUDA']['compiler']['id'] == 'NVIDIA':
                with pytest.raises(RuntimeError, match = 'The host binary file contains more than one embedded CUDA binary file.'):
                    cuobjdump.symtab # pylint: disable=pointless-statement

            # Extract the embedded CUDA binary file and check that calling its symtab works.
            cuobjdump, _ = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                    object_file = False,
                ),
                sass = False,
            )

            assert self.SYMBOL in cuobjdump.symtab['name'].values

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    class TestMany:
        """
        When there are many kernels.

        .. note::

            ``__device__`` functions have been inlined.
        """
        CUDA_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'many.cu'
        CPP_FILE  : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'many.cpp'

        SYMBOLS : typing.Final[dict[str, str]] = {
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
                object_file = True,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump = CuObjDump(file = output, arch = parameters.arch, sass = True)

            TestCuObjDump.dump(
                file = output.with_suffix(f'.{parameters.arch.compute_capability}.sass'),
                cuobjdump = cuobjdump,
            )

            assert len(cuobjdump.functions) == len(self.SYMBOLS)

        def test_extract_cubin_from_file(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the `cubin` from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = False,
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
                    object_file = False,
                ),
            )

            assert cubin.is_file()

            assert set(cuobjdump.functions.keys()) == set(self.SYMBOLS.values())

        def test_extract_symbol_table(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, and extract the symbol table from it.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = False,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, _ = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                    object_file = False,
                ),
                sass = False,
            )

            assert all(x in cuobjdump.symtab['name'].values for x in self.SYMBOLS)

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    class TestCuBLAS:
        """
        Play with the cuBLAS shared library.
        """
        def test_extract_functions(self, parameters : Parameters, workdir : pathlib.Path, cmake_file_api : cmake.FileAPI) -> None:
            """
            Extract a subset of all functions from a randomly picked cubin.

            .. note::

                Sometimes, ``cuobjdump`` will still dump the SASS of unwanted functions.
                That is, even if `--function=...` is passed N mangled function names, it might
                output the SASS for more than N functions.

            .. note::

                The symbol table of cubin files from cuBLAS may contain internal CUDA runtime
                helper functions with symbols such as `$__internal_11_$__cuda_sm20_div_u64`.

                While these symbols appear in the ELF symbol table as valid STT_FUNC,
                ``cuobjdump`` cannot dump any SASS code for them.

                This test therefore filters out any function whose name starts with `$__internal`.
            """
            cublas = CuBLAS(cmake_file_api = cmake_file_api)
            try:
                [cubin] = cublas.extract(arch = parameters.arch, cwd = workdir, randomly = True)
            except IndexError:
                pytest.skip(f'{cublas} does not contain an embedded cubin for {parameters.arch}.')

            symbols = CuObjDump(file = cubin, arch = parameters.arch, sass = False).symtab

            functions = symbols[
                (symbols['type'] == 'STT_FUNC') &
                (~symbols['name'].str.startswith('$__internal'))
            ]

            if functions['name'].shape[0] == 0:
                pytest.skip(f'There is no function or other executable code in {cubin} for {parameters.arch}.')

            chosen = random.sample(functions['name'].values.tolist(), min(functions['name'].shape[0], 3))

            cuobjdump = CuObjDump(file = cubin, arch = parameters.arch, sass = True, keep = chosen)

            assert len(chosen) <= len(cuobjdump.functions.keys()) < symbols.shape[0]

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
