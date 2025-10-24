import abc
import pathlib

from reprospect.tools.architecture import NVIDIAArch
from reprospect.test.environment   import EnvironmentAwareMixin
from reprospect.test.cmake         import CMakeMixin

class TestCase(EnvironmentAwareMixin, abc.ABC):
    """
    Test case.
    """
    @property
    @abc.abstractmethod
    def cwd(self) -> pathlib.Path:
        """
        Working directory for the analysis.
        """

    @property
    @abc.abstractmethod
    def arch(self) -> NVIDIAArch:
        """
        NVIDIA architecture for the analysis.
        """

    @property
    @abc.abstractmethod
    def executable(self) -> pathlib.Path:
        """
        Executable for the analysis.
        """

class CMakeAwareTestCase(CMakeMixin, TestCase):
    """
    Test case with CMake integration.
    """
