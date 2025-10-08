function(add_one_test FILE)

    add_test(
        NAME    tests_python_test_${FILE}
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/python/test/test_${FILE}.py
                    -vvv -s --log-cli-level=info
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    test_environment(tests_python_test_${FILE} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR}/python)
    test_environment(tests_python_test_${FILE} EXECUTABLE set $<TARGET_FILE:tests_cpp_cuda_${FILE}>)
    test_environment(tests_python_test_${FILE} COMPILE_COMMANDS set ${CMAKE_BINARY_DIR}/compile_commands.json)

endfunction()

add_one_test(graph)
add_one_test(saxpy)

test_environment(tests_python_test_graph CMAKE_CUDA_COMPILER_ID set ${CMAKE_CUDA_COMPILER_ID})
