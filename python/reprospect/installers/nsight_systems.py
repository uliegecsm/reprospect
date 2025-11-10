import argparse
import logging
import os
import re
import shutil
import subprocess
import typing

import semantic_version

import system_helpers.apt.install

PACKAGE : typing.Final[str] = 'cuda-nsight-systems'

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

def install(*, args = argparse.Namespace) -> None:
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
    def convert(version : re.Match) -> str:
        logging.info(f"Detected CUDA version is {version.group(0)}.")
        return f'{version.group(1)}-{version.group(2)}'

    if 'CUDA_VERSION' in os.environ:
        version = semantic_version.Version(os.environ['CUDA_VERSION'])
        return f'{version.major}-{version.minor}'
    elif shutil.which('nvidia-smi') is not None:
        output = subprocess.check_output(('nvidia-smi', '--query')).decode()
        for line in output.splitlines():
            if "CUDA Version" in line and (matched := re.search(pattern = r'([0-9]+).([0-9]+)', string = line)) is not None:
                return convert(version = matched)
    elif shutil.which('nvcc') is not None:
        output = subprocess.check_output(('nvcc', '--version')).decode()
        if (matched := re.search(pattern = r'release ([0-9]+).([0-9]+)', string = output)) is not None:
            return convert(version = matched)
    raise RuntimeError("Could not deduce CUDA version.")

def main() -> None:

    logging.basicConfig(level = logging.INFO)

    args = parse_args()
    logging.info(f"Received arguments: {args}")

    install(args = args)

if __name__ == "__main__":

    main()
