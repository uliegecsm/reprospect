import logging
import os
import pathlib

from reprospect.utils import cmake

class TestFileAPI:
    """
    Tests for :py:class:`reprospect.utils.cmake.FileAPI`.
    """
    def test_cache(self) -> None:
        """
        Check that cache variables are read correctly.
        """
        reader = cmake.FileAPI(build_path = pathlib.Path(os.environ['CMAKE_BINARY_DIR']), inspect = {'cache' : 2})

        assert reader.cache['ReProspect_ENABLE_TESTS'].name  == 'ReProspect_ENABLE_TESTS'
        assert reader.cache['ReProspect_ENABLE_TESTS'].value == 'ON'

    def test_toolchains(self) -> None:
        """
        Check toolchain information.
        """
        reader = cmake.FileAPI(build_path = pathlib.Path(os.environ['CMAKE_BINARY_DIR']), inspect = {'toolchains' : 1})

        assert 'CUDA' in reader.toolchains
        assert 'CXX'  in reader.toolchains

        assert reader.toolchains['CUDA'].id in ['NVIDIA', 'Clang']

        for language, compiler in reader.toolchains.items():
            logging.info(f'For language {language}:')
            logging.info(f'\t- compiler ID     : {compiler.id}')
            logging.info(f'\t- compiler path   : {compiler.path}')
            logging.info(f'\t- compiler version: {compiler.version}')
