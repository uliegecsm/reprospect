import shutil

import pytest

from reprospect.tools.binaries import CuppFilt, LlvmCppFilt


class TestCuppFilt:
    """
    Test :py:class:`reprospect.tools.binaries.CuppFilt`.
    """
    def test_demangle(self):
        assert CuppFilt.demangle(s='_Z5saxpyfPKfPfj') == 'saxpy(float, const float *, float *, unsigned int)'

@pytest.mark.skipif(
    shutil.which(LlvmCppFilt.get_executable()) is None, reason=f'requires that {LlvmCppFilt.get_executable()} is installed',
)
class TestLlvmCppFilt:
    """
    Test :py:class:`reprospect.tools.binaries.LlvmCppFilt`.
    """
    def test_demangle(self):
        MANGLED = '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj'
        assert LlvmCppFilt.demangle(s=MANGLED) == 'void add_and_increment_kernel<0u>(unsigned int*)'
        # cu++filt cannot demangle this symbol.
        assert CuppFilt.demangle(s=MANGLED).startswith('_Z')
