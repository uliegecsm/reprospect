function(add_one_example FILE)

    cmake_parse_arguments(
        aoe_args
        ""
        ""
        "EXTRA_PRIVATE_LINK_LIBS;EXTRA_TEST_ENV"
        ${ARGN}
    )

    set(FILE_NAME example_${FILE}.cpp)
    set(FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME})

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILE_PATH})

    set(EXECUTABLE ${gprr_PARENT_RELPATH_REPR}_${FILE})

    add_executable(${EXECUTABLE})

    target_sources(${EXECUTABLE} PRIVATE ${FILE_PATH})

    set_source_files_properties(${FILE_PATH} PROPERTIES LANGUAGE CUDA)

    target_include_directories(${EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR})

    if(aoe_args_EXTRA_PRIVATE_LINK_LIBS)
        target_link_libraries(${EXECUTABLE} PRIVATE ${aoe_args_EXTRA_PRIVATE_LINK_LIBS})
    endif()

    add_test(
        NAME ${EXECUTABLE}
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/example_${FILE}.py
                    --config-file=${CMAKE_SOURCE_DIR}/tests/.pytest.ini
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    pytest_skip_mark(${EXECUTABLE})

    test_environment(${EXECUTABLE} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR})
    test_environment(${EXECUTABLE} CMAKE_BINARY_DIR set ${CMAKE_BINARY_DIR})
    test_environment(${EXECUTABLE} CMAKE_CURRENT_BINARY_DIR set ${CMAKE_CURRENT_BINARY_DIR})

    if(aoe_args_EXTRA_TEST_ENV)
        list(GET aoe_args_EXTRA_TEST_ENV 0 ENV_NAME)
        list(GET aoe_args_EXTRA_TEST_ENV 1 ENV_OP)
        list(GET aoe_args_EXTRA_TEST_ENV 2 ENV_VALUE)
        test_environment(${EXECUTABLE} ${ENV_NAME} ${ENV_OP} ${ENV_VALUE})
    endif()

endfunction()

add_subdirectory(kokkos)
