function(add_one_example FILE)

    set(FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/example_${FILE}.cpp)

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILENAME})

    set(EXECUTABLE ${gprr_PARENT_RELPATH_REPR}_${FILE})

    add_executable(${EXECUTABLE})

    target_sources(${EXECUTABLE} PRIVATE ${FILENAME})

    set_source_files_properties(${FILENAME} PROPERTIES LANGUAGE CUDA)

    target_link_libraries(${EXECUTABLE} PRIVATE Kokkos::kokkoscore)

    add_test(
        NAME ${EXECUTABLE}
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/example_${FILE}.py
                    --config-file=${CMAKE_SOURCE_DIR}/tests/python/.pytest.ini
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    pytest_skip_mark(${EXECUTABLE})

    test_environment(${EXECUTABLE} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR}/python)
    test_environment(${EXECUTABLE} CMAKE_BINARY_DIR set ${CMAKE_BINARY_DIR})
    test_environment(${EXECUTABLE} CMAKE_CURRENT_BINARY_DIR set ${CMAKE_CURRENT_BINARY_DIR})

endfunction()

find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

find_package(
    KokkosTools
    REQUIRED
    COMPONENTS kp_nvtx_connector
)

add_subdirectory(view)
