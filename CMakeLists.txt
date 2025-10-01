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
include(cmake/modules/Warnings.cmake)

add_subdirectory(cpp)

if(ReProspect_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ReProspect_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

#---- Install.
install(FILES LICENSE README.md DESTINATION .)

#---- Package.
set(CPACK_PACKAGE_NAME "reprospect")
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
