import abc
import pathlib

from reprospect.test.cmake import CMakeMixin
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries import CuppFilt, LlvmCppFilt


class TestCase(abc.ABC):
    """
    Test case.
    """
    @property
    @abc.abstractmethod
    def cwd(self) -> pathlib.Path:
        """
        Working directory.
        """

    @property
    @abc.abstractmethod
    def arch(self) -> NVIDIAArch:
        """
        NVIDIA architecture.
        """

    @property
    @abc.abstractmethod
    def executable(self) -> pathlib.Path:
        """
        Executable.
        """

    @property
    @abc.abstractmethod
    def demangler(self) -> type[CuppFilt | LlvmCppFilt]:
        """
        Demangler.
        """

class CMakeAwareTestCase(CMakeMixin, TestCase):
    """
    Test case with CMake integration.
    """
