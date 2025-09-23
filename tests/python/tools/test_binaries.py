import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
import typing

import pytest
import semantic_version
import typeguard

from cuda_helpers.tools.architecture import NVIDIAArch
from cuda_helpers.tools.binaries import get_arch_from_compile_command, CuObjDump, CuppFilt, ResourceUsage

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

@typeguard.typechecked
def get_compilation_output(*, source : pathlib.Path, cwd : pathlib.Path, arch : NVIDIAArch, object : bool = True) -> typing.Tuple[pathlib.Path, str]:
    """
    Compile the `source` in `cwd` for `arch`.
    """
    output = cwd / (source.stem + ('.o' if object else ''))

    cmd = [
        os.environ.get('CMAKE_CUDA_COMPILER_LAUNCHER'),
        os.environ.get('CMAKE_CUDA_COMPILER'),
        '-std=c++20',
    ]

    # Clang tends to add a lot of debug code otherwise.
    cmd.append('-O3')

    match os.environ['CMAKE_CUDA_COMPILER_ID']:
        case 'NVIDIA':
            cmd += [f'-arch={arch.as_sm}', '--resource-usage']
        case 'Clang':
            cmd += [f'--cuda-gpu-arch={arch.as_sm}', '-Xcuda-ptxas', '-v',]
        case _:
            raise ValueError(f'unsupported compiler ID {os.environ['CMAKE_CUDA_COMPILER_ID']}')

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

PARAMETERS = [
    Parameters(arch = NVIDIAArch.from_compute_capability(cc=80)),
    Parameters(arch = NVIDIAArch.from_compute_capability(cc=86)),
    Parameters(arch = NVIDIAArch.from_compute_capability(cc=90)),
    Parameters(arch = NVIDIAArch.from_compute_capability(cc=100)),
    Parameters(arch = NVIDIAArch.from_compute_capability(cc=120)),
]
if semantic_version.Version(os.environ['CUDA_VERSION']) < semantic_version.Version('13.0.0'):
    PARAMETERS.append(Parameters(arch = NVIDIAArch.from_compute_capability(cc=70)))

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestResourceUsage:
    """
    All tests related to reading resource usage from compilation output.
    """
    TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

    class TestSharedMemory:
        """
        When the kernel uses shared memory.
        """
        FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'shared_memory.cu'

        @typeguard.typechecked
        def test(self, parameters : Parameters) -> None:
            """
            Check how shared memory is reported in the compilation output.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = TestResourceUsage.TMPDIR,
                arch = parameters.arch,
                object = True,
            )

            assert f"Compiling entry function '_Z20shared_memory_kernelPfj' for '{parameters.arch.as_sm}'" in compilation

            if parameters.arch.compute_capability < 90:
                assert f'Used 10 registers, used 1 barriers, {128 * 4} bytes smem, 364 bytes cmem[0]' in compilation, compilation
            else:
                assert f'Used 10 registers, used 1 barriers, {128 * 4} bytes smem' in compilation, compilation

    class TestWideLoadStore:
        """
        When the kernel uses wide loads and stores.
        """
        FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'wide_load_store.cu'

        @typeguard.typechecked
        def test(self, parameters : Parameters) -> None:
            """
            Check how wide loads and stores influence the compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = TestResourceUsage.TMPDIR,
                arch = parameters.arch,
                object = True,
            )

            assert f"Compiling entry function '_Z15wide_load_storeP15MyAlignedStructIdEPKS0_' for '{parameters.arch.as_sm}'" in compilation

            if parameters.arch.compute_capability < 90:
                assert 'Used 12 registers, used 0 barriers, 368 bytes cmem[0]' in compilation, compilation
            elif parameters.arch.compute_capability < 100:
                expt_regs = {
                    'NVIDIA' : 14,
                    'Clang' : 12,
                }[os.environ['CMAKE_CUDA_COMPILER_ID']]
                assert f'Used {expt_regs} registers, used 0 barriers' in compilation, compilation
            else:
                assert 'Used 12 registers, used 0 barriers' in compilation, compilation

    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.
        """
        FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'

        @typeguard.typechecked
        def test(self, parameters : Parameters) -> None:
            """
            Check compilation result.
            """
            _, compilation = get_compilation_output(
                source = self.FILE,
                cwd = TestResourceUsage.TMPDIR,
                arch = parameters.arch,
                object = True,
            )

            assert f"Compiling entry function '_Z5saxpyfPKfPfj' for '{parameters.arch.as_sm}'" in compilation

            if parameters.arch.compute_capability < 90:
                assert 'Used 10 registers, used 0 barriers, 380 bytes cmem[0]' in compilation, compilation
            else:
                assert 'Used 10 registers, used 0 barriers' in compilation, compilation

class TestCuppFilt:
    """
    Test :py:class:`cuda_helpers.tools.binaries.CuppFilt`.
    """
    def test_demangle(self):
        assert CuppFilt.demangle(s = '_Z5saxpyfPKfPfj') == 'saxpy(float, const float *, float *, unsigned int)'

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestCuObjDump:
    """
    Tests related to :py:class:`cuda_helpers.tools.binaries.CuObjDump`.
    """
    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.
        """
        FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'
        SIGNATURE = 'saxpy(float, const float *, float *, unsigned int)'

        @typeguard.typechecked
        def test(self, parameters : Parameters) -> None:
            """
            Check compilation result.
            """
            output, compilation = get_compilation_output(
                source = self.FILE,
                cwd = TestResourceUsage.TMPDIR,
                arch = parameters.arch,
                object = True,
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
