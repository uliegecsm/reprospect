add_library(cuda_helpers INTERFACE)
target_include_directories(cuda_helpers INTERFACE ${CMAKE_SOURCE_DIR}/cpp)

add_subdirectory(errors)
