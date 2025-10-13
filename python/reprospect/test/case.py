import functools
import json
import logging
import os
import pathlib
import typing

import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import get_arch_from_compile_command

class EnvironmentAwareMeta(type):
    """
    Metaclass that resolves missing class attributes from the environment.
    """
    @typeguard.typechecked
    def __getattr__(cls, name : str) -> typing.Any:
        return cls.read_from_env(name)

class EnvironmentAware(metaclass = EnvironmentAwareMeta):
    """
    Base class that resolves missing class or instance attributes from the environment.
    """
    #: Specify convertors from environment variable values to attributes of the relevant types.
    _TYPES: dict[str, typing.Callable[[str], typing.Any]] = {}

    @typeguard.typechecked
    @classmethod
    def read_from_env(cls, name : str) -> typing.Any:
        value = os.getenv(name)
        if value is None:
            raise AttributeError(f"{cls.__name__} has no attribute '{name}' and there is no environment variable {name!r}.")

        if name in cls._TYPES:
            value = cls._TYPES[name](value)

        setattr(cls, name, value)

        return value

    @typeguard.typechecked
    def __getattr__(self, name : str) -> typing.Any:
        return self.read_from_env(name)

class TestCase(EnvironmentAware):
    """
    Base class for carrying out analyses of a given executable.

    The child class must define `NAME` (see :py:meth:`cwd`).
    """
    _TYPES = {
        'CMAKE_BINARY_DIR' : pathlib.Path,
        'CMAKE_CURRENT_BINARY_DIR' : pathlib.Path,
    }

    @functools.cached_property
    @typeguard.typechecked
    def cwd(cls) -> pathlib.Path:
        """
        The working directory for the analysis.
        """
        cwd = cls.CMAKE_CURRENT_BINARY_DIR / (cls.NAME + '-case')
        cwd.mkdir(parents = False, exist_ok = True)
        return cwd

    @functools.cached_property
    @typeguard.typechecked
    def arch(cls) -> NVIDIAArch:
        """
        Retrieve the GPU architecture from the compile command database.

        We assume the target file was compiled for only a single architecture.

        See also https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html.
        """
        with open(cls.CMAKE_BINARY_DIR / 'compile_commands.json', 'r') as fin:
            commands = json.load(fin)
        logging.info(f'Looking for {cls.TARGET_SOURCE} in {commands}')
        archs = get_arch_from_compile_command(cmd = next(filter(
            lambda x: str(cls.TARGET_SOURCE) in x['file'],
            commands))['command']
        )
        assert len(archs) == 1
        return archs.pop()
