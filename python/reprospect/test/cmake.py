import abc
import functools
import logging
import json
import pathlib
import typing

import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import get_arch_from_compile_command
from reprospect.utils              import cmake

class CMakeMixin(abc.ABC):
    """
    Mixin to integrate with CMake build system.
    """
    _REQUIRED_ATTRIBUTES_WITH_DEFAULT_FROM_ENV = {
        'CMAKE_BINARY_DIR': pathlib.Path,
        'CMAKE_CURRENT_BINARY_DIR': pathlib.Path,
    }

    @classmethod
    @abc.abstractmethod
    def get_target_name(cls) -> str:
        pass

    @functools.cached_property
    @typeguard.typechecked
    def cwd(self) -> pathlib.Path:
        """
        Get working directory for the analysis based on the CMake current binary directory.
        """
        cwd = self.CMAKE_CURRENT_BINARY_DIR / (self.get_target_name() + '-case')
        cwd.mkdir(parents = False, exist_ok = True)
        return cwd

    @functools.cached_property
    @typeguard.typechecked
    def arch(self) -> NVIDIAArch:
        """
        Retrieve the NVIDIA architecture from the CMake compile command database.

        We assume the target file was compiled for only a single architecture.

        See also https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html.
        """
        with open(self.CMAKE_BINARY_DIR / 'compile_commands.json', 'r', encoding = 'utf-8') as fin:
            commands = json.load(fin)
        target_source = next(self.target_sources)
        logging.info(f'Looking for {target_source} in {commands}')
        archs = get_arch_from_compile_command(cmd = next(filter(
            lambda x: str(target_source) in x['file'],
            commands))['command']
        )
        assert len(archs) == 1
        return archs.pop()

    @functools.cached_property
    @typeguard.typechecked
    def cmake_file_api(self) -> cmake.FileAPI:
        return cmake.FileAPI(cmake_build_directory = self.CMAKE_BINARY_DIR)

    @functools.cached_property
    @typeguard.typechecked
    def target(self) -> dict:
        """
        Retrieve the target information from the CMake codemodel database.
        """
        return self.cmake_file_api.target(self.get_target_name())

    @functools.cached_property
    @typeguard.typechecked
    def target_sources(self) -> typing.Iterator[pathlib.Path]:
        """
        Retrieve the target source from the CMake codemodel database.
        """
        return map(lambda source: pathlib.Path(source['path']), self.target['sources'])

    @functools.cached_property
    @typeguard.typechecked
    def executable(self) -> pathlib.Path:
        """
        Retrieve the executable for the analysis from the CMake codemodel database.
        """
        return self.CMAKE_BINARY_DIR / self.target['paths']['build'] / self.target['nameOnDisk']

    @functools.cached_property
    @typeguard.typechecked
    def toolchains(self) -> dict:
        """
        Retrieve the toolchains information from the read CMake file API.
        """
        return self.cmake_file_api.toolchains
