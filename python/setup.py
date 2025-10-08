import pathlib
import re

from setuptools import setup

version = pathlib.Path(__file__).parent / 'reprospect' / '__init__.py'
version = re.search(r'__version__ = \'([0-9.]+)\'', version.read_text()).group(1)

requirements = pathlib.Path(__file__).parent / 'requirements.txt'
requirements = filter(None, requirements.read_text().splitlines())

setup(
    name             = 'reprospect',
    version          = version,
    license          = 'MIT',
    url              = 'https://github.com/uliegecsm/reprospect',
    install_requires = [
        *requirements,
    ],
    packages = [
        'reprospect.installers',
        'reprospect.test',
        'reprospect.tools',
        'reprospect.utils',
    ],
    package_dir = {
        'reprospect.installers' : 'reprospect/installers',
        'reprospect.test' : 'reprospect/test',
        'reprospect.tools' : 'reprospect/tools',
        'reprospect.utils' : 'reprospect/utils',
    },
    entry_points = {
        'console_scripts': [
            'reprospect-install-nsight-systems = reprospect.installers.nsight_systems:main',
        ],
    },
)
