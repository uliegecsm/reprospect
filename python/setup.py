import pathlib
import re

from setuptools import setup

import cuda_helpers

requirements = pathlib.Path(__file__).parent / 'requirements.txt'
requirements = filter(None, requirements.read_text().splitlines())

setup(
    name             = 'cuda-helpers',
    version          = cuda_helpers.__version__,
    license          = 'MIT',
    url              = 'https://github.com/uliegecsm/cuda-helpers',
    install_requires = [
        *requirements,
    ],
    packages = [
        'cuda_helpers.installers',
        'cuda_helpers.tools',
    ],
    package_dir = {
        'cuda_helpers.installers' : 'cuda_helpers/installers',
        'cuda_helpers.tools' : 'cuda_helpers/tools',
    },
    entry_points = {
        'console_scripts': [
            'cuda-helpers-install-nsight-systems = cuda_helpers.installers.nsight_systems:main',
        ],
    },
)
