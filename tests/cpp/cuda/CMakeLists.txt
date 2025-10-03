function(add_one_test FILE)

    add_executable(tests_cpp_cuda_${FILE} test_${FILE}.cpp)

    target_link_libraries(tests_cpp_cuda_${FILE} PRIVATE CUDA::cudart CUDA::nvtx3)

    set_source_files_properties(test_${FILE}.cpp PROPERTIES LANGUAGE CUDA)

    add_test(NAME tests_cpp_cuda_${FILE} COMMAND $<TARGET_FILE:tests_cpp_cuda_${FILE}>)

endfunction()

find_package(CUDAToolkit REQUIRED COMPONENTS cudart nvtx3)

add_one_test(graph)
add_one_test(saxpy)
