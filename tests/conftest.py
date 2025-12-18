import os
import pathlib

import pytest

from reprospect.utils import cmake


@pytest.fixture(scope = 'session')
def bindir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_BINARY_DIR'])

@pytest.fixture(scope = 'session')
def cmake_file_api(bindir) -> cmake.FileAPI:
    return cmake.FileAPI(cmake_build_directory = bindir)

@pytest.fixture(scope = 'session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])
