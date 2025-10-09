import argparse
import copy
import dataclasses
import json
import logging
import pprint
import typing

import typeguard

from reprospect.tools.architecture import NVIDIAArch

@dataclasses.dataclass
class Compiler:
    ID : str
    version : typing.Optional[str] = None
    path : typing.Optional[str] = None

@typeguard.typechecked
def full_image(*, name : str, tag : str, args : argparse.Namespace) -> str:
    """
    Full image from its `name` and `tag`, with remote.
    """
    return f'{args.registry}/{args.repository}/{name}:{tag}'

@typeguard.typechecked
def complete_job(partial : dict, args : argparse.Namespace) -> dict:
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

    # Name and tag of the image.
    name = 'cuda-' + '-'.join(list(dict.fromkeys([partial['compilers']['CXX'].ID, partial['compilers']['CXX'].version, partial['compilers']['CUDA'].ID])))

    tag = f'{partial['cuda_version']}-devel-ubuntu24.04'

    partial['nvidia_arch'] = str(NVIDIAArch.from_compute_capability(cc = partial['nvidia_compute_capability']))

    partial['base_image'] = f'nvidia/cuda:{tag}'
    partial[     'image'] = full_image(name = name,             tag = tag,                                       args = args)
    partial[    'kokkos'] = full_image(name = f'{name}-kokkos', tag = f'{tag}-{partial['nvidia_arch']}'.lower(), args = args)

    partial['build_platforms'] = ','.join(['linux/amd64'] + partial['additional_build_platforms'] if 'additional_build_platforms' in partial else [])

    # Labels for testing runners.
    runs_on = ['self-hosted', 'linux', 'docker', 'amd64', partial['nvidia_arch'].lower(), 'gpu:0']
    partial['runs-on'] = runs_on

    # Testing is opt-out.
    if 'tests' not in partial:
        partial['tests'] = True

    # Write compilers as dictionaries.
    for lang in partial['compilers']:
        partial['compilers'][lang] = dataclasses.asdict(partial['compilers'][lang])

    return partial

@typeguard.typechecked
def main(*, args : argparse.Namespace) -> None:
    """
    Generate the strategy matrix.
    """
    matrix = []
    environment = {'REGISTRY' : args.registry}

    matrix.append(complete_job({
        'cuda_version' : '12.8.1',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '13'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 70,
        'additional_build_platforms' : ['linux/arm64'],
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '13.0.0',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '14'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 120,
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '13.0.0',
        'compilers' : {'CXX' : Compiler(ID = 'gnu', version = '14'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 86,
        'tests' : False,
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '12.8.1',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '19'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 70,
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '13.0.0',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '19'), 'CUDA' : Compiler(ID = 'nvidia')},
        'nvidia_compute_capability' : 120,
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '12.8.1',
        'compilers' : {'CXX' : Compiler(ID = 'clang', version = '21')},
        'nvidia_compute_capability' : 120,
    }, args = args))

    logging.info(f'Strategy matrix:\n{pprint.pformat(matrix)}')

    # All jobs in the matrix build an image.
    print(f'matrix_images={json.dumps(matrix, default = str)}')

    # But some jobs in the matrix don't require running the tests, because we don't have resources for them.
    print(f'matrix_tests={json.dumps([x for x in matrix if x['tests']], default = str)}')

    print(f'environment={json.dumps(environment, default = str)}')

    print(f'deploy_image={matrix[0]['image']}')

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--registry', type = str, required = True)
    parser.add_argument('--repository', type = str, required = True)

    main(args = parser.parse_args())
