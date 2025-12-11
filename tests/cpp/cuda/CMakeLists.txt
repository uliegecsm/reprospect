function(add_one_executable FILE)

    set(FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/test_${FILE}.cpp)

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILENAME})

    set(EXECUTABLE ${gprr_PARENT_RELPATH_REPR}_${FILE})

    add_executable(${EXECUTABLE})

    target_sources(${EXECUTABLE} PRIVATE ${FILENAME})

    target_link_libraries(${EXECUTABLE} PRIVATE CUDA::cudart CUDA::nvtx3)

    set_source_files_properties(${FILENAME} PROPERTIES LANGUAGE CUDA)

endfunction()

find_package(
    CUDAToolkit REQUIRED
    COMPONENTS cuda_driver cudart nvtx3
)

add_one_executable(graph)
add_one_executable(saxpy)
