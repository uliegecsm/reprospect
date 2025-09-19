# We need the Cuda runtime.
find_package(CUDAToolkit REQUIRED COMPONENTS cudart)

add_executable(tests_cuda_saxpy test_saxpy.cpp)
target_link_libraries(tests_cuda_saxpy PRIVATE CUDA::cudart)
set_source_files_properties(test_saxpy.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(tests_cuda_saxpy PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME tests_cuda_saxpy COMMAND $<TARGET_FILE:tests_cuda_saxpy>)
