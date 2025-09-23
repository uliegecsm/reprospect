cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Retrieve the version.
file(READ "${CMAKE_SOURCE_DIR}/python/cuda_helpers/__init__.py" VERSION_FILE_CONTENTS)
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

#---- Install.
install(FILES LICENSE README.md DESTINATION .)

#---- Package.
set(CPACK_PACKAGE_NAME "cuda-helpers")
set(CPACK_PACKAGE_VENDOR "uliegecsm")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CPACK_ARCHIVE_COMPONENT_INSTALL YES)
set(CPACK_GENERATOR TGZ)

include(CPack)
include(CPackComponent)

cpack_add_component(cpp REQUIRED DISPLAY_NAME "C++ library")
