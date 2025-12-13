import logging
import os
import pathlib

import pytest

from reprospect.utils import cmake

class TestFileAPI:
    """
    Tests for :py:class:`reprospect.utils.cmake.FileAPI`.
    """
    @pytest.fixture(scope = 'session')
    def cmake_file_api(self) -> cmake.FileAPI:
        return cmake.FileAPI(
            cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
        )

    def test_cache(self, cmake_file_api : cmake.FileAPI) -> None:
        """
        Check that cache variables are read correctly.
        """
        assert cmake_file_api.cache['ReProspect_ENABLE_TESTS'] == {
            'properties' : [
                {'name' : 'HELPSTRING', 'value' : 'Enable tests.'},
            ],
            'type' : 'BOOL',
            'value' : 'ON',
        }

    def test_toolchains(self, cmake_file_api : cmake.FileAPI) -> None:
        """
        Check toolchain information.
        """
        assert 'CUDA' in cmake_file_api.toolchains
        assert 'CXX'  in cmake_file_api.toolchains

        assert cmake_file_api.toolchains['CUDA']['compiler']['id'] in ['NVIDIA', 'Clang']

        for language, toolchain in cmake_file_api.toolchains.items():
            compiler = toolchain['compiler']
            logging.info(f'For language {language}:')
            logging.info(f"\t- compiler ID     : {compiler['id']}")
            logging.info(f"\t- compiler path   : {compiler['path']}")
            logging.info(f"\t- compiler version: {compiler['version']}")

    def test_codemodel_configuration(self, cmake_file_api : cmake.FileAPI) -> None:
        """
        Check codemodel configuration information.
        """
        assert cmake_file_api.codemodel_configuration['name'] == 'Release'

        assert len(cmake_file_api.codemodel_configuration['projects']) == 1
        assert cmake_file_api.codemodel_configuration['projects'][0]['name'] == 'reprospect'

        assert 'targets' in cmake_file_api.codemodel_configuration

    def test_target(self, cmake_file_api : cmake.FileAPI) -> None:
        """
        Check target information.
        """
        for name in ['tests_assets_graph', 'tests_assets_saxpy']:
            target = cmake_file_api.target(name = name)

            assert target['name'] == name
            assert 'paths' in target
            assert 'build'  in target['paths']
            assert 'source' in target['paths']
            assert 'nameOnDisk' in target
            assert 'sources' in target
            assert len(target['sources']) == 1
            assert 'path' in target['sources'][0]

        with pytest.raises(ValueError, match = 'Target \'some-random-name\' not found.'):
            cmake_file_api.target(name = 'some-random-name')
