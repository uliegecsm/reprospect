import argparse
import logging
import os
import re
import shutil
import subprocess

import semantic_version
import typeguard

import system_helpers.apt.install

PACKAGE = 'cuda-nsight-systems'

@typeguard.typechecked
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

@typeguard.typechecked
def detect_cuda_version() -> str:
    """
    Detect `Cuda` version using the following strategies:
        1. `CUDA_VERSION` environment variable
        2. `nvidia-smi`
        3. `nvcc`
    """
    def convert(version : str) -> str:
        logging.info(f"Detected Cuda version is {version.group(0)}.")
        version = f'{version.group(1)}-{version.group(2)}'
        return version

    if 'CUDA_VERSION' in os.environ:
        version = semantic_version.Version(os.environ['CUDA_VERSION'])
        return f'{version.major}-{version.minor}'
    elif shutil.which('nvidia-smi') is not None:
        output = subprocess.check_output(['nvidia-smi', '--query']).decode()
        for line in output.splitlines():
            if "CUDA Version" in line:
                return convert(version = re.search(pattern = r'([0-9]+).([0-9]+)', string = line))
    elif shutil.which('nvcc') is not None:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        return convert(version = re.search(pattern = r'release ([0-9]+).([0-9]+)', string = output))
    else:
        raise RuntimeError("Could not deduce Cuda version.")

def main() -> None:

    logging.basicConfig(level = logging.INFO)

    args = parse_args()
    logging.info(f"Received arguments: {args}")

    install(args = args)

if __name__ == "__main__":

    main()
