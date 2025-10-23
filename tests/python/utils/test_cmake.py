import logging
import os
import pathlib

import pytest
import typeguard

from reprospect.utils import cmake

class TestFileAPI:
    """
    Tests for :py:class:`reprospect.utils.cmake.FileAPI`.
    """
    @pytest.fixture(scope = 'session')
    @typeguard.typechecked
    def cmake_file_api(self) -> cmake.FileAPI:
        return cmake.FileAPI(
            build_path = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
        )

    def test_cache(self, cmake_file_api) -> None:
        """
        Check that cache variables are read correctly.
        """
        assert 'name' not in cmake_file_api.cache['ReProspect_ENABLE_TESTS']
        assert cmake_file_api.cache['ReProspect_ENABLE_TESTS']['value'] == 'ON'

    def test_toolchains(self, cmake_file_api) -> None:
        """
        Check toolchain information.
        """
        assert 'CUDA' in cmake_file_api.toolchains
        assert 'CXX'  in cmake_file_api.toolchains

        assert cmake_file_api.toolchains['CUDA']['compiler']['id'] in ['NVIDIA', 'Clang']

        for language, toolchain in cmake_file_api.toolchains.items():
            compiler = toolchain['compiler']
            logging.info(f'For language {language}:')
            logging.info(f'\t- compiler ID     : {compiler['id']}')
            logging.info(f'\t- compiler path   : {compiler['path']}')
            logging.info(f'\t- compiler version: {compiler['version']}')
