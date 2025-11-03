import json
import os
import pathlib

import pytest

from reprospect.tools.architecture    import NVIDIAArch
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
