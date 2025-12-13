import pathlib
import typing

import pytest

from reprospect.utils import cmake

from tests.compilation import get_compilation_output
from tests.parameters  import Parameters, PARAMETERS

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
class TestResourceUsage:
    """
    All tests related to reading resource usage from compilation output.
    """
    class TestSharedMemory:
        """
        When the kernel uses shared memory.
        """
        FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'tools' / 'binaries' / 'assets' / 'shared_memory.cu'

        def test(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check how shared memory is reported in the compilation output.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = True,
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
        FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'tools' / 'binaries' / 'assets' / 'wide_load_store.cu'

        def test(self, workdir : pathlib.Path, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check how wide loads and stores influence the compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = True,
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
        FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'tools' / 'assets' / 'saxpy.cu'

        def test(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Check compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = True,
                resource_usage = True,
                cmake_file_api = cmake_file_api,
            )

            assert f"Compiling entry function '_Z12saxpy_kernelfPKfPfj' for '{parameters.arch.as_sm}'" in compilation, compilation

            if parameters.arch.compute_capability < 90:
                assert 'Used 10 registers, used 0 barriers, 380 bytes cmem[0]' in compilation, compilation
            else:
                assert 'Used 10 registers, used 0 barriers' in compilation, compilation
