import pathlib

from setuptools import setup

import reprospect

requirements = pathlib.Path(__file__).parent / 'requirements.txt'
requirements = filter(None, requirements.read_text().splitlines())

setup(
    name             = 'reprospect',
    version          = reprospect.__version__,
    license          = 'MIT',
    url              = 'https://github.com/uliegecsm/reprospect',
    install_requires = [
        *requirements,
    ],
    packages = [
        'reprospect.installers',
        'reprospect.tools',
    ],
    package_dir = {
        'reprospect.installers' : 'reprospect/installers',
        'reprospect.tools' : 'reprospect/tools',
    },
    entry_points = {
        'console_scripts': [
            'reprospect-install-nsight-systems = reprospect.installers.nsight_systems:main',
        ],
    },
)
