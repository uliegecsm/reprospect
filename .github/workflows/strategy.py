import argparse
import copy
import dataclasses
import json
import logging
import pprint
import typing

from reprospect.tools.architecture import ComputeCapability, NVIDIAArch

COMPILER_ID = typing.Literal['Clang', 'GNU', 'NVIDIA']
LANGUAGE = typing.Literal['CUDA', 'CXX']
PLATFORM = typing.Literal['linux/amd64', 'linux/arm64']

AVAILABLE_RUNNER_ARCHES: typing.Final[tuple[NVIDIAArch, ...]] = (
    NVIDIAArch.from_str('VOLTA70'),
    NVIDIAArch.from_str('BLACKWELL120'),
)

SELF_HOSTED: typing.Final[tuple[str, ...]] = ('self-hosted', 'linux', 'docker', 'amd64')

KOKKOS_SHA: typing.Final[str] = '5.0.0'

@dataclasses.dataclass(frozen=False, slots=True)
class Compiler:
    ID: COMPILER_ID
    version: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if self.path is None:
            match self.ID:
                case 'GNU':
                    self.path = 'g++'
                case 'Clang':
                    self.path = 'clang++'
                case 'NVIDIA':
                    self.path = 'nvcc'
                case _:
                    raise ValueError

JobDict: typing.TypeAlias = dict[str, typing.Any]

@dataclasses.dataclass(frozen=False, slots=True)
class Config:
    cuda_version: str
    ubuntu_version: str
    compilers: dict[LANGUAGE, Compiler]
    compute_capability: ComputeCapability
    platforms: tuple[PLATFORM, ...]

    def __post_init__(self) -> None:
        if 'CUDA' not in self.compilers:
            self.compilers['CUDA'] = copy.deepcopy(self.compilers['CXX'])

        match self.compilers['CUDA'].ID:
            case 'NVIDIA':
                self.compilers['CUDA'].version = self.cuda_version
            case 'Clang':
                pass
            case _:
                raise ValueError

    def jobs(self) -> typing.Generator[JobDict, None, None]:
        """
        Each platform is a separate job, because multi-arch builds are too slow due to emulation.
        """
        for platform in self.platforms:
            yield {
                'cuda_version': self.cuda_version,
                'ubuntu_version': self.ubuntu_version,
                'compilers': self.compilers,
                'compute_capability': self.compute_capability,
                'platform': platform,
            }

def get_base_name_tag_digest(cuda_version: str, ubuntu_version: str) -> tuple[str, str, str]:
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
    return ('nvcr.io/nvidia/cuda', tag, digest)

def full_image(*, name: str, tag: str, platform: PLATFORM, args: argparse.Namespace) -> str:
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
            raise ValueError(platform)

def complete_job_impl(*, partial: JobDict, args: argparse.Namespace) -> JobDict:
    """
    Add fields to a job.
    """
    logging.info(f'Completing job {partial}.')

    # CMake preset (lower case).
    partial['cmake_preset'] = ('-'.join(list(dict.fromkeys([partial['compilers']['CXX'].ID, partial['compilers']['CUDA'].ID])))).lower()

    # We always compile for the 'real' CUDA architecture, see also
    # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html.
    partial['cmake_cuda_architectures'] = f"{partial['compute_capability'].as_int}-real"

    # Kokkos SHA.
    partial['kokkos_sha'] = KOKKOS_SHA

    # Name and tag of the image (lower case).
    name = ('-'.join((
        'cuda',
        *list(dict.fromkeys([
            partial['compilers']['CXX'].ID,
            partial['compilers']['CXX'].version,
            partial['compilers']['CUDA'].ID,
        ])),
    ))).lower()

    base_name, base_tag, base_digest = get_base_name_tag_digest(cuda_version=partial['cuda_version'], ubuntu_version=partial['ubuntu_version'])

    arch = NVIDIAArch.from_compute_capability(cc=partial.pop('compute_capability'))
    partial['nvidia_arch'] = str(arch)

    partial['base_image'] = f'{base_name}:{base_tag}@{base_digest}'
    partial[     'image'] = full_image(platform=partial['platform'], args=args, name=name,                          tag=base_tag)
    partial[    'kokkos'] = full_image(platform=partial['platform'], args=args, name=f'{name}-kokkos-{KOKKOS_SHA}', tag=f'{base_tag}-{arch}'.lower())

    # Write compilers as dictionaries.
    for lang in partial['compilers']:
        partial['compilers'][lang] = dataclasses.asdict(partial['compilers'][lang])
        partial['compilers'][lang]['ID_lower'] = partial['compilers'][lang]['ID'].lower()

    # Environment of the job.
    partial['environment'] = {'REGISTRY': args.registry}

    # Specifics to the 'tests' and 'examples' jobs.
    # Testing is opt-out.
    # We only test for 'linux/amd64'.
    if ('tests' not in partial or partial['tests']) and partial['platform'] == 'linux/amd64':
        partial['tests'   ] = {'container': {'image': partial['image']}}
        partial['examples'] = {'container': {'image': partial['kokkos']}}

        if arch in AVAILABLE_RUNNER_ARCHES:
            # Enforce GPU compute capability. See also
            # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#nvidia-require-constraints.
            env = {'NVIDIA_REQUIRE_ARCH': f'arch={arch.compute_capability.major}.{arch.compute_capability.minor}'}
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
        # All test jobs are still self-hosted.
        # Example jobs for which we don't have a runner available run on the provider runners.
        if arch not in AVAILABLE_RUNNER_ARCHES:
            partial['tests'   ]['runs-on'] = SELF_HOSTED
            partial['examples']['runs-on'] = ('ubuntu-latest',)
        else:
            runs_on = SELF_HOSTED + (f'{arch}'.lower(), 'gpu:0')
            partial['tests'   ]['runs-on'] = runs_on
            partial['examples']['runs-on'] = runs_on
    else:
        for name in ('tests', 'examples'):
            partial[name] = None

    partial['build-images'] = {
        'runs-on': ['self-hosted', 'linux', 'docker', partial['platform'].split('/')[1]],
    }

    return partial

def from_config(config: Config, args: argparse.Namespace) -> list[JobDict]:
    return [
        complete_job_impl(partial=copy.deepcopy(job), args=args)
        for job in config.jobs()
    ]

def main(*, args: argparse.Namespace) -> None:
    """
    Generate the strategy matrix.
    """
    matrix = []

    matrix.extend(from_config(Config(
        cuda_version='12.8.1',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='GNU', version='13'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=7, minor=0),
        platforms=('linux/amd64', 'linux/arm64'),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='12.8.1',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='GNU', version='14'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=9, minor=0),
        platforms=('linux/amd64',),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='13.1.0',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='GNU', version='14'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=12, minor=0),
        platforms=('linux/amd64', 'linux/arm64'),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='12.6.3',
        ubuntu_version='22.04',
        compilers={'CXX': Compiler(ID='GNU', version='12'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=7, minor=5),
        platforms=('linux/amd64',),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='12.6.3',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='GNU', version='13'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=8, minor=9),
        platforms=('linux/amd64',),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='13.0.0',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='GNU', version='14'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=8, minor=6),
        platforms=('linux/amd64',),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='12.8.1',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='Clang', version='19'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=7, minor=0),
        platforms=('linux/amd64',),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='13.1.0',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='GNU', version='14'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=10, minor=0),
        platforms=('linux/amd64',),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='13.0.0',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='Clang', version='20'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=12, minor=0),
        platforms=('linux/amd64',),
    ), args=args))

    matrix.extend(from_config(Config(
        cuda_version='12.8.1',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='Clang', version='21')},
        compute_capability=ComputeCapability(major=12, minor=0),
        platforms=('linux/amd64',),
    ), args=args))

    logging.info(f'Strategy matrix:\n{pprint.pformat(matrix)}')

    # All jobs in the matrix build an image.
    print(f"matrix_images={json.dumps(matrix, default = str)}")

    # But some jobs in the matrix don't require running the tests, because we don't have resources for them.
    print(f"matrix_tests={json.dumps([x for x in matrix if x['tests']], default = str)}")

    print(f"deploy_image={matrix[0]['image']}")
    print(f"doc_image={matrix[0]['kokkos']}")

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--registry', type=str, required=True)
    parser.add_argument('--repository', type=str, required=True)

    main(args=parser.parse_args())
