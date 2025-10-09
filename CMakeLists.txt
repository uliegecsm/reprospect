cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

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

find_package(Python 3.12 REQUIRED Interpreter)

include(cmake/functions/environment.cmake)
include(cmake/functions/path_helper.cmake)
include(cmake/modules/Warnings.cmake)

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
