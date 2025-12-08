import abc
import functools
import json
import pathlib
import typing

from reprospect.test.environment      import EnvironmentField
from reprospect.tools.architecture    import NVIDIAArch
from reprospect.tools.binaries        import CuppFilt, LlvmCppFilt
from reprospect.utils                 import cmake
from reprospect.utils.compile_command import get_arch_from_compile_command

def get_demangler_for_compiler(compiler_id : str) -> type[CuppFilt | LlvmCppFilt]:
    """
    Get demangler for compiler with given id.
    """
    match compiler_id:
        case 'NVIDIA':
            return CuppFilt
        case 'Clang':
            return LlvmCppFilt
        case _:
            raise ValueError(f'unsupported compiler ID {compiler_id}')

class CMakeMixin(abc.ABC):
    """
    Mixin to integrate with CMake build system.
    """
    CMAKE_BINARY_DIR = EnvironmentField(converter = pathlib.Path)
    CMAKE_CURRENT_BINARY_DIR = EnvironmentField(converter = pathlib.Path)

    @classmethod
    @abc.abstractmethod
    def get_target_name(cls) -> str:
        pass

    @functools.cached_property
    def cwd(self) -> pathlib.Path:
        """
        Get working directory for the analysis based on the CMake current binary directory.
        """
        cwd = self.CMAKE_CURRENT_BINARY_DIR / (self.get_target_name() + '-case')
        cwd.mkdir(parents = False, exist_ok = True)
        return cwd

    @functools.cached_property
    def arch(self) -> NVIDIAArch:
        """
        Retrieve the NVIDIA architecture from the CMake compile command database.

        We assume the target file was compiled for only a single architecture.

        See also https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html.
        """
        with open(self.CMAKE_BINARY_DIR / 'compile_commands.json', encoding = 'utf-8') as fin:
            commands = json.load(fin)
        target_source = next(self.target_sources)
        archs = get_arch_from_compile_command(cmd = next(filter(
            lambda x: str(target_source) in x['file'],
            commands))['command'],
        )
        assert len(archs) == 1
        return archs.pop()

    @functools.cached_property
    def cmake_file_api(self) -> cmake.FileAPI:
        return cmake.FileAPI(cmake_build_directory = self.CMAKE_BINARY_DIR)

    @functools.cached_property
    def target(self) -> cmake.TargetDict:
        """
        Retrieve the target information from the CMake codemodel database.
        """
        return self.cmake_file_api.target(self.get_target_name())

    @functools.cached_property
    def target_sources(self) -> typing.Generator[pathlib.Path, None, None]:
        """
        Retrieve the target source from the CMake codemodel database.
        """
        return (pathlib.Path(source['path']) for source in self.target['sources'])

    @functools.cached_property
    def executable(self) -> pathlib.Path:
        """
        Retrieve the executable for the analysis from the CMake codemodel database.
        """
        return self.CMAKE_BINARY_DIR / self.target['paths']['build'] / self.target['nameOnDisk']

    @functools.cached_property
    def toolchains(self) -> cmake.ToolchainDict:
        """
        Retrieve the toolchains information from the read CMake file API.
        """
        return self.cmake_file_api.toolchains

    @functools.cached_property
    def demangler(self) -> type[CuppFilt | LlvmCppFilt]:
        return get_demangler_for_compiler(self.toolchains['CUDA']['compiler']['id'])
