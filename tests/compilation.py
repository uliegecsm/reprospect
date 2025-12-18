import logging
import pathlib
import subprocess

from reprospect.tools.architecture import NVIDIAArch
from reprospect.utils import cmake

QUALITY_FLAGS: tuple[str, ...] = ('-Wall', '-Wextra', '-Werror')

def get_compilation_output(*, # pylint: disable=too-many-branches
    source: pathlib.Path,
    cwd: pathlib.Path,
    arch: NVIDIAArch,
    cmake_file_api: cmake.FileAPI,
    object_file: bool = True,
    resource_usage: bool = False,
    ptx: bool = False,
) -> tuple[pathlib.Path, str]:
    """
    Compile the `source` in `cwd` for `arch`.

    :param object_file: Whether to compile into an object file.
    """
    cuda_toolchain = cmake_file_api.toolchains['CUDA']
    cuda_compiler_id = cuda_toolchain['compiler']['id']

    output = cwd / (source.stem + '.' + arch.as_sm + ('.o' if object_file else ''))

    cmd = [
        cmake_file_api.cache['CMAKE_CUDA_COMPILER_LAUNCHER']['value'],
        cuda_toolchain['compiler']['path'],
        '-std=c++20',
    ]

    # Code quality flags.
    match cuda_compiler_id:
        case 'NVIDIA':
            cmd.extend(f'-Xcompiler={x}' for x in QUALITY_FLAGS)
        case 'Clang':
            cmd.extend(QUALITY_FLAGS)
            cmd.append('-Wno-error=unknown-cuda-version')
        case _:
            raise ValueError(f'unsupported compiler ID {cuda_compiler_id}')

    # For compiling an executable, if the source ends with '.cpp', we need '-x cu' for nvcc and '-x cuda' for clang.
    if not object_file and source.suffix == '.cpp':
        match cuda_compiler_id:
            case 'NVIDIA':
                cmd += ['-x', 'cu']
            case 'Clang':
                cmd += ['-x', 'cuda']
            case _:
                raise ValueError(f"unsupported compiler ID {cuda_compiler_id}")

    # Clang tends to add a lot of debug code otherwise.
    cmd.append('-O3')

    match cuda_compiler_id:
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
                cmd += ['-Xcuda-ptxas', '-v']
        case _:
            raise ValueError(f"unsupported compiler ID {cuda_compiler_id}")

    # Allow any include within the repository.
    cmd.append(f"-I{cmake_file_api.cache['reprospect_SOURCE_DIR']['value']}")

    # Append link flags if needed.
    if not object_file:
        cmd.append(f"-L{pathlib.Path(cmake_file_api.cache['CUDA_cudart_LIBRARY']['value']).parent}")
        cmd.append('-lcudart')
        cmd.append(f"-L{pathlib.Path(cmake_file_api.cache['CUDA_cuda_driver_LIBRARY']['value']).parent}")
        cmd.append('-lcuda')
    else:
        cmd.append('-c')

    cmd.extend((
        source,
        '-o', output,
    ))

    logging.info(f'Compiling {source} with {cmd} in {cwd}.')

    return (output, subprocess.check_output(
        args = cmd,
        cwd = cwd,
        stderr = subprocess.STDOUT,
    ).decode())

def get_cubin_name(*, compiler_id: str, file: pathlib.Path, arch: NVIDIAArch, object_file: bool = False) -> str:
    """
    When the compilation is *not* into an object file, the resulting file may contain:

    * more than one (usually 2) embedded CUDA binary files when using ``nvcc``
    * only one when using ``clang``

    For ``nvcc``, the first cubin is usually nearly empty.
    """
    if object_file:
        return f'{file.stem}.1.{arch.as_sm}.cubin'

    match compiler_id:
        case 'NVIDIA':
            return f'{file.stem}.2.{arch.as_sm}.cubin'
        case 'Clang':
            return f'{file.stem}.1.{arch.as_sm}.cubin'
        case _:
            raise ValueError(f'unsupported compiler ID {compiler_id}')
