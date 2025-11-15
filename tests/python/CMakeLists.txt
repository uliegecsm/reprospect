function(add_pytest FILE)

    set(FILEPATH ${CMAKE_CURRENT_SOURCE_DIR}/test_${FILE}.py)

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILEPATH})

    set(FILENAME ${gprr_PARENT_RELPATH_REPR}_${FILE})

    add_test(
        NAME ${FILENAME}
        COMMAND ${Python_EXECUTABLE} -m pytest
                    --typeguard-packages=reprospect
                    --config-file=${CMAKE_SOURCE_DIR}/tests/python/.pytest.ini
                    ${FILEPATH}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    test_environment(${FILENAME} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR}/python)
    test_environment(${FILENAME} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR})
    test_environment(${FILENAME} CMAKE_BINARY_DIR set ${CMAKE_BINARY_DIR})
    test_environment(${FILENAME} CMAKE_CURRENT_BINARY_DIR set ${CMAKE_CURRENT_BINARY_DIR})

    pytest_skip_mark(${FILENAME})

    # Directory for artifacts.
    file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${FILEPATH})
    test_environment(${FILENAME} ARTIFACT_DIR set ${CMAKE_SOURCE_DIR}/artifacts/${relative_path})

endfunction()

add_pytest(compilation)

add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(utils)
