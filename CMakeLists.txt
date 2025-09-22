cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Retrieve the version.
file(READ "${CMAKE_SOURCE_DIR}/cuda_helpers/__init__.py" VERSION_FILE_CONTENTS)
string(REGEX MATCH "__version__ = '([^']+)'" _match "${VERSION_FILE_CONTENTS}")

# Define the project.
project(
    cuda-helpers
    VERSION "${CMAKE_MATCH_1}"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.12 REQUIRED Interpreter)

include(cmake/functions/environment.cmake)
include(cmake/modules/Warnings.cmake)

add_subdirectory(cpp)

enable_testing()

add_subdirectory(tests)
