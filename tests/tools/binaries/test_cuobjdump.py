import logging
import pathlib
import random
import typing

import pytest

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump, CuppFilt, Function, ResourceUsage
from reprospect.utils              import cmake
from reprospect.utils              import rich_helpers

from tests.compilation import get_compilation_output, get_cubin_name
from tests.cublas      import CuBLAS
from tests.parameters  import Parameters, PARAMETERS

class TestResourceUsage:
    """
    Tests related to :py:class:`reprospect.tools.binaries.cuobjdump.ResourceUsage`.
    """
    def test(self) -> None:
        assert ResourceUsage.parse('REG:61 STACK:0 SHARED:8448 LOCAL:0 CONSTANT[0]:452 TEXTURE:0 SURFACE:0 SAMPLER:0') == ResourceUsage(
            register=61, shared=8448,
            constant={0: 452},
        )
        assert ResourceUsage.parse('REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0') == ResourceUsage(
            register=32,
            constant={0: 392},
        )
        assert ResourceUsage.parse('REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[2]:128 CONSTANT[0]:392 TEXTURE:1 SURFACE:2 SAMPLER:3') == ResourceUsage(
            register=32,
            constant={0: 392, 2: 128},
            texture=1, surface=2, sampler=3,
        )

        with pytest.raises(ValueError):
            ResourceUsage.parse('not-a-valid-resource-usage')

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    class TestSharedMemory:
        FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'shared_memory.cu'
        SIGNATURE: typing.Final[str] = '_Z20shared_memory_kernelPfj'

        def test(self, workdir: pathlib.Path, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
            output, compilation = get_compilation_output(
                source=self.FILE,
                cwd=workdir,
                arch=parameters.arch,
                object_file=True,
                resource_usage=True,
                cmake_file_api=cmake_file_api,
            )

            assert f"Compiling entry function '{self.SIGNATURE}' for '{parameters.arch.as_sm}'" in compilation, compilation

            ru = CuObjDump(file=output, arch=parameters.arch, demangler=None).functions[self.SIGNATURE].ru

            if parameters.arch.compute_capability < 90:
                assert f'Used 10 registers, used 1 barriers, {128 * 4} bytes smem, 364 bytes cmem[0]' in compilation, compilation
                assert ru == ResourceUsage(register=10, shared=128 * 4, constant={0: 364})
            elif parameters.arch.compute_capability < 100:
                assert f'Used 10 registers, used 1 barriers, {128 * 4} bytes smem' in compilation, compilation
                assert ru == ResourceUsage(register=10, shared=1536, constant={0: 540})
            else:
                assert f'Used 10 registers, used 1 barriers, {128 * 4} bytes smem' in compilation, compilation
                assert ru == ResourceUsage(register=10, shared=1536, constant={0: 908})

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    class TestWideLoadStore:
        FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'wide_load_store.cu'
        SIGNATURE: typing.Final[str] = '_Z22wide_load_store_kernelP15MyAlignedStructIdEPKS0_'

        def test(self, workdir: pathlib.Path, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
            output, compilation = get_compilation_output(
                source=self.FILE,
                cwd=workdir,
                arch=parameters.arch,
                object_file=True,
                resource_usage=True,
                cmake_file_api=cmake_file_api,
            )

            assert f"Compiling entry function '{self.SIGNATURE}' for '{parameters.arch.as_sm}'" in compilation, compilation

            ru = CuObjDump(file=output, arch=parameters.arch, demangler=None).functions[self.SIGNATURE].ru

            if parameters.arch.compute_capability < 90:
                assert 'Used 12 registers, used 0 barriers, 368 bytes cmem[0]' in compilation, compilation
                assert ru == ResourceUsage(register=12, constant={0: 368})
            elif parameters.arch.compute_capability < 100:
                expt_regs = {
                    'NVIDIA': 14,
                    'Clang': 12,
                }[cmake_file_api.toolchains['CUDA']['compiler']['id']]
                assert f'Used {expt_regs} registers, used 0 barriers' in compilation, compilation
                assert ru == ResourceUsage(register=expt_regs, constant={0: 544})
            else:
                assert 'Used 12 registers, used 0 barriers' in compilation, compilation
                assert ru == ResourceUsage(register=12, constant={0: 912})

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    class TestSaxpy:
        FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent.parent / 'tools' / 'assets' / 'saxpy.cu'
        SIGNATURE: typing.Final[str] = '_Z12saxpy_kernelfPKfPfj'

        def test(self, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
            output, compilation = get_compilation_output(
                source=self.FILE,
                cwd=workdir,
                arch=parameters.arch,
                object_file=True,
                resource_usage=True,
                cmake_file_api=cmake_file_api,
            )

            assert f"Compiling entry function '{self.SIGNATURE}' for '{parameters.arch.as_sm}'" in compilation, compilation

            ru = CuObjDump(file=output, arch=parameters.arch, demangler=None).functions[self.SIGNATURE].ru

            if parameters.arch.compute_capability < 90:
                assert 'Used 10 registers, used 0 barriers, 380 bytes cmem[0]' in compilation, compilation
                assert ru == ResourceUsage(register=10, constant={0: 380})
            elif parameters.arch.compute_capability < 100:
                assert 'Used 10 registers, used 0 barriers' in compilation, compilation
                assert ru == ResourceUsage(register=10, constant={0: 556})
            else:
                assert 'Used 10 registers, used 0 barriers' in compilation, compilation
                assert ru == ResourceUsage(register=10, constant={0: 924})

class TestFunction:
    """
    Tests related to :py:class:`reprospect.tools.binaries.Function`.
    """
    SYMBOL: typing.Final[str] = '_Z9my_kernelfPKfPfj'

    CODE: typing.Final[str] = """\
        .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
        /*0000*/                   LDC R1, c[0x0][0x37c]                &wr=0x0          ?trans1;           /* 0x0000df00ff017b82 */
                                                                                                            /* 0x000e220000000800 */
        /*0010*/                   S2R R0, SR_TID.X                     &wr=0x1          ?trans7;           /* 0x0000000000007919 */
                                                                                                            /* 0x000e6e0000002100 */
"""

    RU: typing.Final[ResourceUsage] = ResourceUsage(
        register=10,
        constant={0: 924},
    )

    def test_string_representation(self) -> None:
        """
        Test :py:meth:`reprospect.tools.binaries.Function.__str__`.
        """
        function = Function(symbol = self.SYMBOL, code = self.CODE, ru = self.RU)

        # Check that the conversion to a table with truncation of long lines works as expected.
        rt = function.to_table(max_code_length = 120)
        assert rich_helpers.to_string(rt) == """\
┌─────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Symbol          │ _Z9my_kernelfPKfPfj                                                                                                      │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
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
│ Symbol          │ _Z9my_kernelfPKfPfj                                                                                                                │
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

class TestCuObjDump:
    """
    Tests related to :py:class:`reprospect.tools.binaries.CuObjDump`.
    """
    @staticmethod
    def dump(*, file: pathlib.Path, cuobjdump: CuObjDump) -> None:
        logging.info(f'Writing parsed SASS to {file}.')
        file.write_text(str(cuobjdump))

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.
        """
        CPP_FILE:  typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cpp'
        CUDA_FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cu'
        SYMBOL:    typing.Final[str] = '_Z12saxpy_kernelfPKfPfj'
        SIGNATURE: typing.Final[str] = CuppFilt.demangle(SYMBOL)

        def test_sass_from_object(self, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
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

            assert cuobjdump.functions[self.SIGNATURE].ru == ResourceUsage(
                register=10,
                constant={0: cst},
            ), cuobjdump.functions[self.SIGNATURE].ru

            assert cuobjdump.functions[self.SIGNATURE].symbol == self.SYMBOL

        def test_extract_cubin_from_file(self, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
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

        def test_extract_symbol_table(self, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
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
                with pytest.raises(RuntimeError, match = r'The host binary file contains more than one embedded CUDA binary file.'):
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
        CUDA_FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'many.cu'
        CPP_FILE:  typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'many.cpp'

        FUNCTIONS: typing.Final[dict[str, str]] = {
            '_Z6say_hiv': 'say_hi()',
            '_Z20vector_atomic_add_42PKfS0_Pfj': 'vector_atomic_add_42(const float *, const float *, float *, unsigned int)',
        }

        def test_sass_from_object(self, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
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

            assert len(cuobjdump.functions) == len(self.FUNCTIONS)

            assert all(cuobjdump.functions[signature].symbol == symbol for symbol, signature in self.FUNCTIONS.items())

        def test_extract_cubin_from_file(self, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
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

            assert set(cuobjdump.functions.keys()) == set(self.FUNCTIONS.values())

        def test_extract_symbol_table(self, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
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

            assert all(symbol in cuobjdump.symtab['name'].values for symbol in self.FUNCTIONS)

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    class TestCuBLAS:
        """
        Play with the cuBLAS shared library.
        """
        def test_extract_functions(self, parameters: Parameters, workdir: pathlib.Path, cmake_file_api: cmake.FileAPI) -> None:
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
            'my_kernel(float, const float *, float *, unsigned int)': Function(
                symbol = TestFunction.SYMBOL,
                code = TestFunction.CODE,
                ru = TestFunction.RU,
            ),
            'my_other_kernel(float, const float *, float *, unsigned int)': Function(
                symbol = '_Z15my_other_kernelfPKfPfj',
                code = TestFunction.CODE,
                ru = TestFunction.RU,
            ),
        }

        assert str(cuobjdump) == """\
CuObjDump of code_object.o for architecture BLACKWELL120:
┌─────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Function        │ my_kernel(float, const float *, float *, unsigned int)                                                                             │
├─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Symbol          │ _Z9my_kernelfPKfPfj                                                                                                                │
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
│ Symbol          │ _Z15my_other_kernelfPKfPfj                                                                                                         │
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
