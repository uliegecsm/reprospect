import json
import pathlib
import typing

import cmake_file_api
import typeguard

class FileAPI:
    """
    Thin wrapper for https://github.com/madebr/python-cmake-file-api.
    """
    @typeguard.typechecked
    def __init__(self, build_path : pathlib.Path, inspect : typing.Dict[str, int]) -> None:
        """
        Use our wrapper for cache variables and toolchain information.
        """
        reader = cmake_file_api.reply.v1.api.CMakeFileApiV1(build_path = build_path)

        for kind, version in inspect.items():
            match kind:
                case 'cache':
                    self.cache = self.inspect_cache(reader = reader, version = version)
                case 'toolchains':
                    self.toolchains = {x.language: x.compiler for x in reader.inspect(kind = cmake_file_api.kinds.kind.ObjectKind(kind), kind_version = version).toolchains}
                case _:
                    setattr(self, kind, reader.inspect(kind = cmake_file_api.kinds.kind.ObjectKind(kind), kind_version = version))

    @typeguard.typechecked
    def inspect_cache(self, reader, version : int) -> dict:
        """
        By default, https://github.com/madebr/python-cmake-file-api/blob/3caf111d1ba10f5f9ae624336b55d3e7ca33f9e6/cmake_file_api/reply/v1/api.py#L70
        will return a list of cache entries, which is impractical for name-based access.
        """
        path = reader._create_reply_path() / reader.index().reply.stateless[(cmake_file_api.kinds.kind.ObjectKind.CACHE, version)].jsonFile # pylint: disable=protected-access

        with path.open() as file:
            dikt = json.load(file)

        if dikt['kind'] != cmake_file_api.kinds.kind.ObjectKind.CACHE.value:
            raise ValueError(f'unexpected kind {dikt['kind']}')

        return {ce.name: ce for ce in map(cmake_file_api.kinds.cache.v2.CacheEntry.from_dict, dikt['entries'])}
