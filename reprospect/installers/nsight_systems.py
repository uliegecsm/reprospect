import argparse
import logging
import os
import re
import shutil
import subprocess
import typing

import semantic_version
import system_helpers.apt.install

from reprospect.utils import nvcc

PACKAGE: typing.Final[str] = 'cuda-nsight-systems'

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Install Nsight Systems with apt.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--version', type = str, required = False, help = f'Version of the package ({PACKAGE}) to install.', default = detect_cuda_version())

    return parser.parse_args()

def install(*, args: argparse.Namespace) -> None:
    """
    Install `Nsight Systems` through `apt`.
    """
    package = PACKAGE + '-' + args.version
    logging.info(f'Installing nsight-systems with {package}.')
    system_helpers.apt.install.install_packages(
        packages = [package],
        update = True, clean = True,
    )

def detect_cuda_version() -> str:
    """
    Detect CUDA version using the following strategies:

    #. ``CUDA_VERSION`` environment variable
    #. ``nvidia-smi --query``
    #. ``nvcc --version``
    """
    if 'CUDA_VERSION' in os.environ:
        version = semantic_version.Version(os.environ['CUDA_VERSION'])
        return f'{version.major}-{version.minor}'

    if shutil.which('nvidia-smi') is not None:
        for line in subprocess.check_output(('nvidia-smi', '--version')).decode().splitlines():
            if "CUDA Version" in line and (matched := re.search(pattern = r'([0-9]+).([0-9]+)', string = line)) is not None:
                return f'{matched.group(1)}-{matched.group(2)}'

    if shutil.which('nvcc') is not None:
        version = nvcc.get_version()
        return f'{version.major}-{version.minor}'

    raise RuntimeError("Could not deduce CUDA version.")

def main() -> None:

    logging.basicConfig(level = logging.INFO)

    args = parse_args()
    logging.info(f"Received arguments: {args}")

    install(args = args)

if __name__ == "__main__":

    main()
