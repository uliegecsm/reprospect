cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Retrieve the version.
file(READ "${CMAKE_SOURCE_DIR}/cuda_helpers/__init__.py" VERSION_FILE_CONTENTS)
string(REGEX MATCH "__version__ = '([^']+)'" _match "${VERSION_FILE_CONTENTS}")

# Define the project.
project(cuda-helpers VERSION "${CMAKE_MATCH_1}")
