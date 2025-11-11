import logging
import pathlib
import subprocess
import typing

from reprospect.utils              import cmake
from reprospect.tools.architecture import NVIDIAArch

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
