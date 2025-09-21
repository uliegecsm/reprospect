import json
import os
import pathlib

from cuda_helpers.tools.architecture import NVIDIAArch
from cuda_helpers.tools.binaries import get_arch_from_compile_command

def test_get_arch_from_compile_command():
    """
    Test :py:meth:`cuda_helpers.tools.binaries.get_arch_from_compile_command`.
    """
    COMMANDS = {
        # For nvcc.
        '-arch=sm_86' : set([NVIDIAArch.from_compute_capability(cc = 86)]),
        '--generate-code=arch=compute_90,code=[compute_90,sm_90]' : set([NVIDIAArch.from_str('HOPPER90')]),
        # For clang.
        '--cuda-gpu-arch=sm_120' : set([NVIDIAArch.from_str('BLACKWELL120')]),
        '--cuda-gpu-arch=sm_120 --cuda-gpu-arch=sm_86' : set([NVIDIAArch.from_str('BLACKWELL120'), NVIDIAArch.from_str('AMPERE86')]),
    }

    for command, arch in COMMANDS.items():
        assert get_arch_from_compile_command(cmd = command) == arch

    with open(pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'compile_commands.json', 'r') as fin:
        compile_commands = json.load(fin)

    for command in filter(lambda x: x['file'].endswith('tests/cuda/test_saxpy.cpp'), compile_commands):
        assert get_arch_from_compile_command(cmd = command['command']) == set([NVIDIAArch.from_compute_capability(cc = os.environ['CMAKE_CUDA_ARCHITECTURES'])])
