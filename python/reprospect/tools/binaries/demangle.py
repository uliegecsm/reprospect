import abc
import pathlib
import subprocess
import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class DemanglerMixin(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_executable(cls) -> pathlib.Path | str:
        pass

    @classmethod
    def demangle(cls, s: str) -> str:
        """
        Demangle `s` (a symbol).
        """
        return subprocess.check_output((
            cls.get_executable(), s,
        )).decode().strip()

class CuppFilt(DemanglerMixin):
    """
    Convenient wrapper for ``cu++filt``.
    """
    @classmethod
    @override
    def get_executable(cls) -> str:
        return 'cu++filt'

class LlvmCppFilt(DemanglerMixin):
    """
    Convenient wrapper for ``llvm-cxxfilt``.
    """
    @classmethod
    @override
    def get_executable(cls) -> str:
        return 'llvm-cxxfilt'
