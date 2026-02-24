import argparse
import json
import logging
import os
import subprocess
import urllib.request

import packaging.version


def get_available(package: str) -> list[packaging.version.Version]:
    """
    Get available versions of `package` from `PyPI`.
    """
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/json') as resp:
        data = json.load(resp)
        return [packaging.version.Version(x) for x in data["releases"]]

def get_candidate(*, target: packaging.version.Version, versions: list[packaging.version.Version]) -> packaging.version.Version | None:
    """
    Match exactly if possible; otherwise, match the closest lower patch version that has the same major and minor.
    """
    candidate = None

    for version in versions:
        # Reject pre/dev/post releases.
        if version.pre or version.dev or version.post:
            continue

        # Exact match.
        if version == target:
            return version

        if version.major == target.major and version.minor == target.minor \
            and version < target and (candidate is None or version > candidate):
            candidate = version
    return candidate

def install_cuda_bindings(target: packaging.version.Version) -> None:
    """
    Pick a version of ``cuda-bindings`` that best matches `target`.

    .. warning::

        ``cuda-bindings`` was once part of the ``cuda-python`` package, which is now a meta-package.
        Therefore, installing ``cuda-bindings`` alone might be impossible for some versions.
        For instance, there is no standalone ``cuda-bindings==12.6``, but this version can be installed
        as part of ``cuda-python==12.6``.

    References:

    * https://nvidia.github.io/cuda-python/latest/
    * https://nvidia.github.io/cuda-python/cuda-bindings/latest/
    """
    versions = get_available(package='cuda-bindings')
    logging.info(f'Available versions for \'cuda-bindings\': {versions}.')

    candidate = get_candidate(target=target, versions=versions)

    if candidate is not None:
        logging.info(f'Installing \'cuda-bindings\' {candidate}.')
        requirement = f'cuda-bindings=={candidate}'
    else:
        logging.warning('Could not find a suitable version of \'cuda-bindings\'.')

        versions = get_available(package='cuda-python')
        logging.info(f'Available versions for \'cuda-python\': {versions}.')

        candidate = get_candidate(target=target, versions=versions)

        if candidate is None:
            raise RuntimeError('Could not find a suitable version of \'cuda-python\'.')

        logging.info(f'Installing \'cuda-python\' {candidate}.')
        requirement = f'cuda-python=={candidate}'

    subprocess.check_call(('pip', 'install', requirement))

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--version', type=packaging.version.Version, required=False, default=os.environ['CUDA_VERSION'])

    return parser.parse_args()

def main() -> None:

    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    logging.info(f'Received arguments: {args}.')

    install_cuda_bindings(target=args.version)

if __name__ == '__main__':

    main()
