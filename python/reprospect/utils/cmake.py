import functools
import pathlib

import cmake_file_api
import ijson
import typeguard

class FileAPI:
    """
    Thin wrapper for https://github.com/madebr/python-cmake-file-api.
    """
    CACHE_VERSION = 2
    TOOLCHAINS_VERSION = 1

    typeguard.typechecked
    def __init__(self, build_path : pathlib.Path) -> None:
        """
        Use our wrapper for cache variables and toolchain information.
        """
        self.reader = cmake_file_api.reply.v1.api.CMakeFileApiV1(build_path = build_path)

    @functools.cached_property
    @typeguard.typechecked
    def cache(self) -> dict:
        cache_json = self.reader._create_reply_path() / self.reader.index().reply.stateless[(cmake_file_api.kinds.kind.ObjectKind.CACHE, self.CACHE_VERSION)].jsonFile

        with cache_json.open('rb') as file:
            entries = ijson.items(file, 'entries.item')
            return {
                entry['name']: {k: v for k, v in entry.items() if k != 'name'}
                for entry in entries
            }

    @functools.cached_property
    @typeguard.typechecked
    def toolchains(self) -> dict:
        toolchains_json = self.reader._create_reply_path() / self.reader.index().reply.stateless[(cmake_file_api.kinds.kind.ObjectKind.TOOLCHAINS, self.TOOLCHAINS_VERSION)].jsonFile

        with toolchains_json.open('rb') as file:
            toolchains = ijson.items(file, 'toolchains.item')
            return {
                toolchain['language']: {k: v for k, v in toolchain.items() if k != 'language'}
                for toolchain in toolchains
            }
