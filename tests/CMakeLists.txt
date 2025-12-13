function(add_pytest FILE)

    set(FILE_NAME test_${FILE}.py)
    set(FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME})

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILE_PATH})

    set(EXECUTABLE ${gprr_PARENT_RELPATH_REPR}_${FILE})

    if(ReProspect_ENABLE_COVERAGE)
        set(LAUNCHER coverage run --source=${CMAKE_SOURCE_DIR} --data-file=${CMAKE_SOURCE_DIR}/coverage/${gprr_PARENT_RELPATH}/${FILE_NAME}.cov)
    else()
        set(LAUNCHER ${Python_EXECUTABLE})
    endif()

    add_test(
        NAME ${EXECUTABLE}
        COMMAND ${LAUNCHER} -m pytest
                    --typeguard-packages=reprospect
                    --config-file=${CMAKE_SOURCE_DIR}/tests/.pytest.ini
                    ${FILE_PATH}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    test_environment(${EXECUTABLE} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR})
    test_environment(${EXECUTABLE} CMAKE_BINARY_DIR set ${CMAKE_BINARY_DIR})
    test_environment(${EXECUTABLE} CMAKE_CURRENT_BINARY_DIR set ${CMAKE_CURRENT_BINARY_DIR})

    pytest_skip_mark(${EXECUTABLE})

    # Directory for artifacts.
    test_environment(${EXECUTABLE} ARTIFACT_DIR set ${CMAKE_SOURCE_DIR}/artifacts/${gprr_PARENT_RELPATH}/${FILE_NAME})

endfunction()

add_pytest(compilation)

add_subdirectory(assets)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(utils)
