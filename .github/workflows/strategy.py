import argparse
import json
import logging
import pprint

from reprospect.tools.architecture import NVIDIAArch

def full_image(*, name : str, tag : str, args : argparse.Namespace) -> str:
    """
    Full image from its `name` and `tag`, with remote.
    """
    return f'{args.registry}/{args.repository}/{name}:{tag}'

def complete_job(partial : dict, args : argparse.Namespace) -> dict:
    """
    Add fields to a job.
    """
    logging.info(f'Completing job {partial}.')

    assert isinstance(partial['nvidia_compute_capability'], int)
    assert all(isinstance(x, tuple) for x in partial['compiler_family'])

    name = 'cuda-' + '-'.join(partial['compiler_family'][0])

    if len(partial['compiler_family']) > 1 and partial['compiler_family'][1][0] == 'nvcc':
        name += '-nvcc'
        partial['compiler_family'][1] = ('nvcc', partial['cuda_version'])

    partial['cmake_preset'] = '-'.join([x[0] for x in partial['compiler_family']])

    tag = f'{partial['cuda_version']}-devel-ubuntu24.04'

    partial['nvidia_arch'] = str(NVIDIAArch.from_compute_capability(cc = partial['nvidia_compute_capability']))

    partial['base_image'] = f'nvidia/cuda:{tag}'
    partial[     'image'] = full_image(name = name, tag = tag, args = args)

    partial['build_platforms'] = ','.join(['linux/amd64'] + partial['additional_build_platforms'] if 'additional_build_platforms' in partial else [])

    # Labels for testing runners.
    runs_on = ['self-hosted', 'linux', 'docker', 'amd64', partial['nvidia_arch'].lower(), 'gpu:0']
    partial['runs-on'] = runs_on

    return partial

def main(*, args : argparse.Namespace) -> None:
    """
    Generate the strategy matrix.
    """
    matrix = []
    environment = {'REGISTRY' : args.registry}

    matrix.append(complete_job({
        'cuda_version' : '12.8.1',
        'compiler_family' : [('gcc', '13'), ('nvcc',)],
        'nvidia_compute_capability' : 70,
        'additional_build_platforms' : ['linux/arm64'],
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '13.0.0',
        'compiler_family' : [('gcc', '14'), ('nvcc',)],
        'nvidia_compute_capability' : 120,
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '12.8.1',
        'compiler_family' : [('clang', '19'), ('nvcc',)],
        'nvidia_compute_capability' : 70,
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '13.0.0',
        'compiler_family' : [('clang', '19'), ('nvcc',)],
        'nvidia_compute_capability' : 120,
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '12.8.1',
        'compiler_family' : [('clang', '21')],
        'nvidia_compute_capability' : 120,
    }, args = args))

    logging.info(f'Strategy matrix:\n{pprint.pformat(matrix)}')

    print(f'matrix={json.dumps(matrix, default = str)}')
    print(f'environment={json.dumps(environment, default = str)}')
    print(f'deploy_image={matrix[0]['image']}')

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--registry', type = str, required = True)
    parser.add_argument('--repository', type = str, required = True)

    main(args = parser.parse_args())
