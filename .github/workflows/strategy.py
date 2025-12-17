import argparse
import copy
import dataclasses
import json
import logging
import pprint
import typing

from reprospect.tools.architecture import NVIDIAArch

AVAILABLE_RUNNER_ARCHES : typing.Final[tuple[NVIDIAArch, ...]] = (
    NVIDIAArch.from_str('VOLTA70'),
    NVIDIAArch.from_str('BLACKWELL120'),
)

JobDict : typing.TypeAlias = dict[str, typing.Any]

def get_base_name_tag_digest(cuda_version : str, ubuntu_version : str) -> tuple[str, str, str]:
    """
    Get `Docker` base image name, tag and digest.
    """
    match (cuda_version, ubuntu_version):
        case ('12.6.3', '22.04'):
            tag = '12.6.3-devel-ubuntu22.04'
            digest = 'sha256:d49bb8a4ff97fb5fe477947a3f02aa8c0a53eae77e11f00ec28618a0bcaa2ad1'
        case ('12.6.3', '24.04'):
            tag = '12.6.3-devel-ubuntu24.04'
            digest = 'sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade'
        case ('12.8.1', '24.04'):
            tag = '12.8.1-devel-ubuntu24.04'
            digest = 'sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66'
        case ('13.0.0', '24.04'):
            tag = '13.0.0-devel-ubuntu24.04'
            digest = 'sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d'
        case ('13.1.0', '24.04'):
            tag = '13.1.0-devel-ubuntu24.04'
            digest = 'sha256:7f32ae6e575abb29f2dacf6c75fe94a262bb48dcc5196ac833ced59d9fde8107'
        case _:
            raise ValueError((cuda_version, ubuntu_version))
    return ('nvidia/cuda', tag, digest)

@dataclasses.dataclass(frozen = False, slots = True)
class Compiler:
    ID : str
    version : str | None = None
    path : str | None = None

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

def complete_job_impl(*, partial : JobDict, args : argparse.Namespace) -> JobDict:
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
            raise ValueError(f"unsupported CXX compiler ID {partial['compilers']['CXX'].ID}")

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
            raise ValueError(f"unsupported CUDA compiler ID {partial['compilers']['CUDA'].ID}")

    # CMake preset.
    partial['cmake_preset'] = '-'.join(list(dict.fromkeys([partial['compilers']['CXX'].ID, partial['compilers']['CUDA'].ID])))

    # We always compile for the 'real' CUDA architecture, see also
    # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html.
    partial['cmake_cuda_architectures'] = f"{partial['nvidia_compute_capability']}-real"

    # Name and tag of the image.
    name = 'cuda-' + '-'.join(list(dict.fromkeys([partial['compilers']['CXX'].ID, partial['compilers']['CXX'].version, partial['compilers']['CUDA'].ID])))

    base_name, base_tag, base_digest = get_base_name_tag_digest(cuda_version = partial['cuda_version'], ubuntu_version = partial['ubuntu_version'])

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

    # Specifics to the 'tests' and 'examples' jobs.
    # Testing is opt-out.
    # We only test for 'linux/amd64'.
    if ('tests' not in partial or partial['tests']) and partial['platform'] == 'linux/amd64':
        partial['tests'   ] = {'container' : {'image' : partial['image']}}
        partial['examples'] = {'container' : {'image' : partial['kokkos']}}

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

        # We do require the architecture as a label if the architecture is part of our
        # available runner fleet.
        runs_on = ['self-hosted', 'linux', 'docker', 'amd64']
        if arch in AVAILABLE_RUNNER_ARCHES:
            runs_on += [f'{arch}'.lower(), 'gpu:0']
        for name in ('tests', 'examples'):
            partial[name]['runs-on'] = runs_on
    else:
        for name in ('tests', 'examples'):
            partial[name] = None

    return partial

def complete_job(partial : JobDict, args : argparse.Namespace) -> list[JobDict]:
    """
    Each platform is a separate job, because multi-arch builds are too slow due to emulation.
    """
    jobs : list[JobDict] = []

    for platform in partial.pop('platforms'):
        job = copy.deepcopy(partial)
        job['platform'] = platform

        job = complete_job_impl(partial = job, args = args)
        job['build-images'] = {
            'runs-on' : ['self-hosted', 'linux', 'docker', platform.split('/')[1]],
        }

        jobs.append(job)

    return jobs

def main(*, args : argparse.Namespace) -> None:
    """
    Generate the strategy matrix.
    """
    matrix = []

    matrix.extend(complete_job({
        'cuda_version' : '12.8.1',
        'ubuntu_version' : '24.04',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '13'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 70,
        'platforms' : ['linux/amd64', 'linux/arm64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '13.1.0',
        'ubuntu_version' : '24.04',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '14'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 120,
        'platforms' : ['linux/amd64', 'linux/arm64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '12.6.3',
        'ubuntu_version' : '22.04',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '12'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 75,
        'platforms' : ['linux/amd64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '12.6.3',
        'ubuntu_version' : '24.04',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '13'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 89,
        'platforms' : ['linux/amd64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '13.0.0',
        'ubuntu_version' : '24.04',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '14'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 86,
        'platforms' : ['linux/amd64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '12.8.1',
        'ubuntu_version' : '24.04',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '19'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 70,
        'platforms' : ['linux/amd64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '13.0.0',
        'ubuntu_version' : '24.04',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '20'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 120,
        'platforms' : ['linux/amd64'],
    }, args = args))

    matrix.extend(complete_job({
        'cuda_version' : '12.8.1',
        'ubuntu_version' : '24.04',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '21')},
        'nvidia_compute_capability' : 120,
        'platforms' : ['linux/amd64'],
    }, args = args))

    logging.info(f'Strategy matrix:\n{pprint.pformat(matrix)}')

    # All jobs in the matrix build an image.
    print(f"matrix_images={json.dumps(matrix, default = str)}")

    # But some jobs in the matrix don't require running the tests, because we don't have resources for them.
    print(f"matrix_tests={json.dumps([x for x in matrix if x['tests']], default = str)}")

    print(f"deploy_image={matrix[0]['image']}")
    print(f"doc_image={matrix[0]['kokkos']}")

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--registry', type = str, required = True)
    parser.add_argument('--repository', type = str, required = True)

    main(args = parser.parse_args())
