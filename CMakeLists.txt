cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

# Retrieve the version.
file(READ "${CMAKE_SOURCE_DIR}/python/reprospect/__init__.py" VERSION_FILE_CONTENTS)
string(REGEX MATCH "__version__ = '([^']+)'" _match "${VERSION_FILE_CONTENTS}")

# Define the project.
project(
    reprospect
    VERSION "${CMAKE_MATCH_1}"
    LANGUAGES CXX CUDA
)

option(ReProspect_ENABLE_TESTS "Enable tests." OFF)
option(ReProspect_ENABLE_EXAMPLES "Enable examples." OFF)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.10 REQUIRED Interpreter)

include(cmake/functions/environment.cmake)
include(cmake/functions/path_helper.cmake)
include(cmake/functions/pytest_skip.cmake)
include(cmake/modules/detect.cmake)
include(cmake/modules/Warnings.cmake)

# Ask CMake to dump file-based detailed information about the build system.
# See also https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html.
# Note that this requires CMake 3.27 or newer.
cmake_file_api(QUERY API_VERSION 1
  CACHE 2
  TOOLCHAINS 1
  CODEMODEL 2
)

if(ReProspect_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ReProspect_ENABLE_EXAMPLES)
    enable_testing()
    add_subdirectory(examples)
endif()

if(ReProspect_ENABLE_DOCS)
    add_subdirectory(docs)
endif()
