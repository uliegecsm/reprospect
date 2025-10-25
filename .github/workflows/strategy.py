import argparse
import copy
import dataclasses
import json
import logging
import pprint
import typing

import typeguard

from reprospect.tools.architecture import NVIDIAArch

AVAILABLE_RUNNER_ARCHES = (
    NVIDIAArch.from_str('VOLTA70'),
    NVIDIAArch.from_str('BLACKWELL120'),
)

@typeguard.typechecked
def get_base_name_tag_digest(version : str) -> tuple[str, str, str]:
    """
    Get `Docker` base image name, tag and digest.
    """
    match version:
        case '12.6.3':
            tag = f'{version}-devel-ubuntu24.04'
            digest = 'sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade'
        case '12.8.1':
            tag = f'{version}-devel-ubuntu24.04'
            digest = 'sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66'
        case '13.0.0':
            tag = f'{version}-devel-ubuntu24.04'
            digest = 'sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d'
        case '13.0.1':
            tag = f'{version}-devel-ubuntu24.04'
            digest = 'sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92'
        case _:
            raise ValueError(version)
    return ('nvidia/cuda', tag, digest)

@dataclasses.dataclass
class Compiler:
    ID : str
    version : typing.Optional[str] = None
    path : typing.Optional[str] = None

@typeguard.typechecked
def full_image(*, name : str, tag : str, platform : str, args : argparse.Namespace) -> str:
    """
    Full image from its `name` and `tag`, with remote.

    For now, `linux/arm64` are suffixed with `-arm64`, and we don't build a manifest for multi-arch images.
    """
    value = f'{args.registry}/{args.repository}/{name}:{tag}'
    match platform:
        case 'linux/amd64':
            return value
        case 'linux/arm64':
            return value + '-arm64'
        case _:
            raise ValueError(f'unsupported platform {platform!r}')

@typeguard.typechecked
def complete_job_impl(*, partial : dict, args : argparse.Namespace) -> dict:
    """
    Add fields to a job.
    """
    logging.info(f'Completing job {partial}.')

    assert isinstance(partial['nvidia_compute_capability'], int)
    assert isinstance(partial['compilers'], dict)
    assert all(k in ['CXX', 'CUDA'] and isinstance(v, Compiler) for k, v in partial['compilers'].items())

    # Complete CXX compiler.
    match partial['compilers']['CXX'].ID:
        case 'gnu':
            partial['compilers']['CXX'].path = 'g++'
        case 'clang':
            partial['compilers']['CXX'].path = 'clang++'
        case _:
            raise ValueError(f'unsupported CXX compiler ID {partial['compilers']['CXX'].ID}')

    # Complete CUDA compiler.
    if 'CUDA' not in partial['compilers']:
        partial['compilers']['CUDA'] = copy.deepcopy(partial['compilers']['CXX'])

    match partial['compilers']['CUDA'].ID:
        case 'nvidia':
            partial['compilers']['CUDA'].path    = 'nvcc'
            partial['compilers']['CUDA'].version = partial['cuda_version']
        case 'clang':
            pass
        case _:
            raise ValueError(f'unsupported CUDA compiler ID {partial['compilers']['CUDA'].ID}')

    # CMake preset.
    partial['cmake_preset'] = '-'.join(list(dict.fromkeys([partial['compilers']['CXX'].ID, partial['compilers']['CUDA'].ID])))

    # We always compile for the 'real' Cuda architecture, see also
    # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html.
    partial['cmake_cuda_architectures'] = f'{partial['nvidia_compute_capability']}-real'

    # Name and tag of the image.
    name = 'cuda-' + '-'.join(list(dict.fromkeys([partial['compilers']['CXX'].ID, partial['compilers']['CXX'].version, partial['compilers']['CUDA'].ID])))

    base_name, base_tag, base_digest = get_base_name_tag_digest(version = partial['cuda_version'])

    arch = NVIDIAArch.from_compute_capability(cc = partial.pop('nvidia_compute_capability'))
    partial['nvidia_arch'] = str(arch)

    partial['base_image'] = f'{base_name}:{base_tag}@{base_digest}'
    partial[     'image'] = full_image(name = name,             tag = base_tag,                     platform = partial['platform'], args = args)
    partial[    'kokkos'] = full_image(name = f'{name}-kokkos', tag = f'{base_tag}-{arch}'.lower(), platform = partial['platform'], args = args)

    # Write compilers as dictionaries.
    for lang in partial['compilers']:
        partial['compilers'][lang] = dataclasses.asdict(partial['compilers'][lang])

    # Environment of the job.
    partial['environment'] = {'REGISTRY' : args.registry}

    # Specifics to the 'tests', 'examples' and 'install-as-package-and-test' jobs.
    # Testing is opt-out.
    # We only test for 'linux/amd64'.
    if ('tests' not in partial or partial['tests']) and partial['platform'] == 'linux/amd64':
        partial['tests'                      ] = {'container' : {'image' : partial['image']}}
        partial['examples'                   ] = {'container' : {'image' : partial['kokkos']}}
        partial['install-as-package-and-test'] = {'container' : {'image' : partial['kokkos']}}

        if arch in AVAILABLE_RUNNER_ARCHES:
            # Enforce GPU compute capability. See also
            # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#nvidia-require-constraints.
            env = {'NVIDIA_REQUIRE_ARCH' : f'arch={arch.compute_capability.major}.{arch.compute_capability.minor}'}
            partial['tests'   ]['container']['env'] = env
            partial['examples']['container']['env'] = env
        else:
            # Any runner can pick up this job, even those with incompatible CUDA drivers.
            # Since no executable will actually run, we allow such runners to proceed.
            # To prevent any GPU-related issues or convoluted conditional skips based on driver version in the tests,
            # we explicitly hide all GPUs.
            env = {'NVIDIA_VISIBLE_DEVICES': ''}
            partial['tests'   ]['container']['env'] = env
            partial['examples']['container']['env'] = env

        # We don't run anything on GPU.
        # See also https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#nvidia-disable-require-environment-variable
        partial['install-as-package-and-test']['container']['env'] = {
            'NVIDIA_DISABLE_REQUIRE' : '1',
        }

        # We do require the architecture as a label if the architecture is part of our
        # available runner fleet.
        runs_on = ['self-hosted', 'linux', 'docker', 'amd64']
        if arch in AVAILABLE_RUNNER_ARCHES:
            runs_on += [f'{arch}'.lower(), 'gpu:0']
        for name in ['tests', 'examples', 'install-as-package-and-test']:
            partial[name]['runs-on'] = runs_on
    else:
        for name in ['tests', 'examples', 'install-as-package-and-test']:
            partial[name] = None

    return partial

@typeguard.typechecked
def complete_job(partial : dict, args : argparse.Namespace) -> list[dict]:
    """
    Each platform is a separate job, because multi-arch builds are too slow due to emulation.
    """
    jobs = []

    for platform in partial.pop('platforms'):
        job = copy.deepcopy(partial)
        job['platform'] = platform

        job = complete_job_impl(partial = job, args = args)
        job['build-images'] = {
            'runs-on' : ['self-hosted', 'linux', 'docker', platform.split('/')[1]],
        }

        jobs.append(job)

    return jobs

@typeguard.typechecked
def main(*, args : argparse.Namespace) -> None:
    """
    Generate the strategy matrix.
    """
    matrix = []

    matrix.extend(complete_job({
        'cuda_version' : '12.8.1',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '13'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 70,
        'platforms' : ['linux/amd64', 'linux/arm64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '13.0.1',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '14'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 120,
        'platforms' : ['linux/amd64', 'linux/arm64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '12.6.3',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '13'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 89,
        'platforms' : ['linux/amd64'],
        'tests' : True,
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '13.0.0',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '14'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 86,
        'platforms' : ['linux/amd64'],
        'tests' : True,
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '12.8.1',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '19'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 70,
        'platforms' : ['linux/amd64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '13.0.0',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '20'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 120,
        'platforms' : ['linux/amd64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '12.8.1',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '21')},
        'nvidia_compute_capability' : 120,
        'platforms' : ['linux/amd64'],
    }, args = args))

    logging.info(f'Strategy matrix:\n{pprint.pformat(matrix)}')

    # All jobs in the matrix build an image.
    print(f'matrix_images={json.dumps(matrix, default = str)}')

    # But some jobs in the matrix don't require running the tests, because we don't have resources for them.
    print(f'matrix_tests={json.dumps([x for x in matrix if x['tests']], default = str)}')

    print(f'deploy_image={matrix[0]['image']}')
    print(f'doc_image={matrix[0]['kokkos']}')

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--registry', type = str, required = True)
    parser.add_argument('--repository', type = str, required = True)

    main(args = parser.parse_args())
