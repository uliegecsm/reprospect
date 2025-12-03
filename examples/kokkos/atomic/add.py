import pathlib
import re
import subprocess
import typing

import pytest

from reprospect.test           import CMakeAwareTestCase
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass     import Decoder
from reprospect.utils          import detect

class TestCase(CMakeAwareTestCase):
    """
    Derived type must to define :py:attr:`SIGNATURE_MATCHER`.
    """
    SIGNATURE_MATCHER : typing.ClassVar[re.Pattern[str]]

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'{self.get_target_name()}.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope = 'class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(
            file = self.executable,
            arch = self.arch,
            sass = True, cwd = self.cwd,
            cubin = self.cubin.name,
            demangler = self.demangler,
        )[0]

    @pytest.fixture(scope = 'class')
    def decoder(self, cuobjdump : CuObjDump) -> Decoder:
        [sig] = [sig for sig in cuobjdump.functions if self.SIGNATURE_MATCHER.search(sig) is not None]
        return Decoder(code = cuobjdump.functions[sig].code)

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test(self) -> None:
        """
        Run the executable.
        """
        subprocess.check_call(self.executable)
