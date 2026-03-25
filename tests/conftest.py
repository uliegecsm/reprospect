import os
import pathlib

import pytest
from cmake_file_api.kinds.toolchains.v1 import CMakeToolchainCompiler

from reprospect.utils import cmake


@pytest.fixture(scope='session')
def bindir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_BINARY_DIR'])

@pytest.fixture(scope='session')
def cmake_file_api(bindir) -> cmake.FileAPI:
    return cmake.FileAPI(cmake_build_directory=bindir)

@pytest.fixture(scope='session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

@pytest.fixture(scope='session')
def cmake_cuda_compiler(cmake_file_api) -> CMakeToolchainCompiler:
    return cmake_file_api.compiler(toolchain='CUDA')
