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
    build_image_examples: bool = True
    enable_tests: bool | None = None
    enable_examples: bool | None = None

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

        if self.build_image_examples is False:
            if self.enable_examples is True:
                raise RuntimeError('Examples enabled, but examples image not built.')
            self.enable_examples = False

    def jobs(self) -> typing.Generator[JobDict, None, None]:
        """
        Each platform is a separate job, because multi-arch builds are too slow due to emulation.
        """
        for platform in self.platforms:
            # We don't run tests and examples for 'linux/arm64'.
            if platform == 'linux/arm64':
                if self.enable_tests is True or self.enable_examples is True:
                    raise RuntimeError(f'Running tests or examples for {platform} is not supported yet.')
                enable_tests = enable_examples = False
            else:
                enable_tests    = self.enable_tests    is not False
                enable_examples = self.enable_examples is not False

            yield {
                'cuda_version': self.cuda_version,
                'ubuntu_version': self.ubuntu_version,
                'compilers': self.compilers,
                'compute_capability': self.compute_capability,
                'platform': platform,
                'build_image_examples': self.build_image_examples,
                'enable_tests': enable_tests,
                'enable_examples': enable_examples,
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

def require_arch(arch: NVIDIAArch) -> dict[str, str]:
    """
    If the `arch` is available in the runner fleet, enforce GPU compute capability with `NVIDIA_REQUIRE_ARCH`.
    See also
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#nvidia-require-constraints.

    Otherwise, any runner can pick up this job, even those with incompatible CUDA drivers.
    Since no executable will actually run, we allow such runners to proceed.
    To prevent any GPU-related issues or convoluted conditional skips based on driver version in the tests,
    we explicitly hide all GPUs.
    """
    if arch in AVAILABLE_RUNNER_ARCHES:
        return {'NVIDIA_REQUIRE_ARCH': f'arch={arch.compute_capability.major}.{arch.compute_capability.minor}'}
    return {'NVIDIA_VISIBLE_DEVICES': ''}

def runs_on(arch: NVIDIAArch, jtype: typing.Literal['tests', 'examples']) -> tuple[str, ...]:
    """
    We do require the architecture as a label if the architecture is part of our
    available runner fleet.

    Example jobs for which we don't have a runner available run on the provider runners.
    """
    if arch in AVAILABLE_RUNNER_ARCHES:
        return SELF_HOSTED + (f'{arch}'.lower(), 'gpu:0')
    match jtype:
        case 'tests':
            return SELF_HOSTED
        case 'examples':
            return ('ubuntu-latest',)
        case _:
            raise ValueError

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

    # Name of the job.
    # See also https://futurestud.io/tutorials/github-actions-customize-the-job-name.
    partial['name'] = ', '.join((
        arch.as_sm,
        partial['cuda_version'],
        partial['compilers']['CXX'].ID,
        partial['compilers']['CXX'].version,
        partial['compilers']['CUDA'].ID,
        partial['ubuntu_version'],
        partial['platform'],
    ))

    # Write compilers as dictionaries.
    for lang in partial['compilers']:
        partial['compilers'][lang] = dataclasses.asdict(partial['compilers'][lang])
        partial['compilers'][lang]['ID_lower'] = partial['compilers'][lang]['ID'].lower()

    # Environment of the job.
    partial['environment'] = {'REGISTRY': args.registry}

    # Specifics to the 'tests' jobs.
    if partial['enable_tests']:
        partial['tests'] = {'container': {'image': partial['image']}}
        partial['tests']['container']['env'] = require_arch(arch=arch)
        partial['tests']['runs-on'] = runs_on(arch=arch, jtype='tests')

    # Specifics to the 'examples' jobs.
    if partial['enable_examples']:
        partial['examples'] = {'container': {'image': partial['kokkos']}}
        partial['examples']['container']['env'] = require_arch(arch=arch)
        partial['examples']['runs-on'] = runs_on(arch=arch, jtype='examples')

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
        cuda_version='13.1.0',
        ubuntu_version='24.04',
        compilers={'CXX': Compiler(ID='Clang', version='21'), 'CUDA': Compiler(ID='NVIDIA')},
        compute_capability=ComputeCapability(major=10, minor=3),
        platforms=('linux/amd64',),
        build_image_examples=False,
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

    print(f"matrix_examples={json.dumps([x for x in matrix if 'examples' in x], default = str)}")
    print(f"matrix_tests={json.dumps([x for x in matrix if 'tests' in x], default = str)}")

    print(f"deploy_image={matrix[0]['image']}")
    print(f"doc_image={matrix[0]['kokkos']}")

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--registry', type=str, required=True)
    parser.add_argument('--repository', type=str, required=True)

    main(args=parser.parse_args())
