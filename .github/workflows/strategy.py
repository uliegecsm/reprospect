import argparse
import json
import logging
import os

def full_image(*, name : str, tag : str, args : argparse.Namespace) -> str:
    return f'{args.registry}/{args.repository}/{name}:{tag}'

def complete_job(partial : dict, args : argparse.Namespace) -> dict:
    logging.info(f'Completing job {partial}.')
    name = 'cuda-' + '-'.join(partial['compiler_family'])
    tag = f'{partial['cuda_version']}-devel-ubuntu24.04'
    partial['base_image'] = f'nvidia/cuda:{tag}'
    partial[     'image'] = full_image(name = name, tag = tag, args = args)
    return partial

def main(*, args : argparse.Namespace) -> None:
    """
    Generate the strategy matrix.
    """
    matrix = []
    environment = {'REGISTRY' : args.registry}

    matrix.append(complete_job({
        'cuda_version' : '12.8.1',
        'compiler_family' : ['gcc', 'nvcc'],
        'cuda_arch' : 'volta70',
    }, args = args))

    matrix.append(complete_job({
        'cuda_version' : '13.0.0',
        'compiler_family' : ['gcc', 'nvcc'],
        'cuda_arch' : 'blackwell120',
    }, args = args))

    # Labels for testing runners.
    for job in matrix:
        runs_on = ['self-hosted', 'linux', 'docker', 'amd64', job['cuda_arch'], 'gpu:0']
        job['runs-on'] = runs_on

    print(f'matrix={json.dumps(matrix, default = str)}')
    print(f'environment={json.dumps(environment, default = str)}')
    print(f'deploy_image={matrix[0]['image']}')

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--registry', type = str, required = True)
    parser.add_argument('--repository', type = str, required = True)

    main(args = parser.parse_args())
