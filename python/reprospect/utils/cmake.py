import functools
import json
import pathlib

import cmake_file_api
import ijson

class FileAPI:
    """
    Retrieve information about the CMake buildsystem.

    References:

    * https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html
    * https://github.com/madebr/python-cmake-file-api.
    """
    CACHE_VERSION = 2
    TOOLCHAINS_VERSION = 1
    CODEMODEL_VERSION = 2

    def __init__(self, cmake_build_directory : pathlib.Path) -> None:
        self.reader = cmake_file_api.reply.v1.api.CMakeFileApiV1(build_path = cmake_build_directory)
        self.cmake_reply_path = cmake_build_directory / '.cmake' / 'api' / 'v1' / 'reply'

    @functools.cached_property
    def cache(self) -> dict:
        """
        Retrieve the CMake cache.
        """
        cache_json = self.cmake_reply_path / self.reader.index().reply.stateless[(cmake_file_api.kinds.kind.ObjectKind.CACHE, self.CACHE_VERSION)].jsonFile

        with cache_json.open('rb') as file:
            entries = ijson.items(file, 'entries.item')
            return {
                entry['name']: {k: v for k, v in entry.items() if k != 'name'}
                for entry in entries
            }

    @functools.cached_property
    def toolchains(self) -> dict:
        """
        Retrieve the toolchains information.
        """
        toolchains_json = self.cmake_reply_path / self.reader.index().reply.stateless[(cmake_file_api.kinds.kind.ObjectKind.TOOLCHAINS, self.TOOLCHAINS_VERSION)].jsonFile

        with toolchains_json.open('rb') as file:
            toolchains = ijson.items(file, 'toolchains.item')
            return {
                toolchain['language']: {k: v for k, v in toolchain.items() if k != 'language'}
                for toolchain in toolchains
            }

    @functools.cached_property
    def codemodel_configuration(self) -> dict:
        """
        Retrieve the codemodel information, and extract the information available for the build configuration.

        This function assumes that there is only a single build configuration.
        """
        codemodel_json = self.cmake_reply_path / self.reader.index().reply.stateless[(cmake_file_api.kinds.kind.ObjectKind.CODEMODEL, self.CODEMODEL_VERSION)].jsonFile

        with codemodel_json.open('rb') as file:
            configurations = list(ijson.items(file, 'configurations.item'))

        assert len(configurations) == 1, "Handling of multiple CMake configurations not implemented."

        return configurations[0]

    @functools.lru_cache(maxsize = 128)
    def target(self, name : str) -> dict:
        """
        Retrieve the information available for the target `name`.
        """
        for target in self.codemodel_configuration['targets']:
            if target['name'] == name:
                with (self.cmake_reply_path / target['jsonFile']).open() as file:
                    return json.load(file)
        raise ValueError(f'Target {name!r} not found.')
