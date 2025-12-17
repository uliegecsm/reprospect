function(add_one_executable FILE)

    set(FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/test_${FILE}.cpp)

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILENAME})

    set(EXECUTABLE ${gprr_PARENT_RELPATH_REPR}_${FILE})

    add_executable(${EXECUTABLE})

    target_sources(${EXECUTABLE} PRIVATE ${FILENAME})

    target_link_libraries(${EXECUTABLE} PRIVATE CUDA::cudart nvtx3::nvtx3-cpp)

    set_source_files_properties(${FILENAME} PROPERTIES LANGUAGE CUDA)

endfunction()

find_package(
    CUDAToolkit REQUIRED
    COMPONENTS cuda_driver cudart
)
find_package(nvtx3 REQUIRED)

add_one_executable(graph)
add_one_executable(half)
add_one_executable(saxpy)

add_test(
    NAME tests_assets_nvtx
    COMMAND nsys profile
                --trace=nvtx
                --sample=none
                --backtrace=none
                --cpuctxsw=none
                ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_nvtx.py
                    --config-file=${CMAKE_SOURCE_DIR}/tests/.pytest.ini
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
test_environment(tests_assets_nvtx PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR})
