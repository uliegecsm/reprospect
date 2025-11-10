import os
import pathlib
import typing

from reprospect.utils              import cmake
from reprospect.tools.architecture import NVIDIAArch

from tests.python.compilation import get_compilation_output

class GemmTensorTile:
    """
    The kernel generates many SASS lines (several hundreds).
    """
    FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).with_suffix('.cu')
    SIGNATURE : typing.Final[str] = 'gemm_tensor_tile(const __half *, const __half *, float *, unsigned int, unsigned int, unsigned int)'

    @classmethod
    def executable(cls, arch : NVIDIAArch, cwd : pathlib.Path, cmake_file_api : cmake.FileAPI) -> pathlib.Path:
        output, _ = get_compilation_output(
            source = cls.FILE,
            cwd = cwd,
            arch = arch,
            object = False,
            resource_usage = True,
            cmake_file_api = cmake_file_api,
            includedirs = (
                cmake_file_api.cache['reprospect_SOURCE_DIR']['value'],
            ),
        )

        assert output.is_file()
        assert os.access(output, os.X_OK)

        return output
