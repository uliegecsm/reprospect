import pathlib
import re

from setuptools import setup

version = pathlib.Path(__file__).parent / 'reprospect' / '__init__.py'
version = re.search(r'__version__ = \'([0-9.]+)\'', version.read_text()).group(1)

requirements = pathlib.Path(__file__).parent / 'requirements.txt'
requirements = filter(None, requirements.read_text().splitlines())

setup(
    version          = version,
    install_requires = [
        *requirements,
    ],
)
