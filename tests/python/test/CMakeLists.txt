function(add_one_test FILE)

    add_test(
        NAME    tests_python_test_${FILE}
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/python/test/test_${FILE}.py
                    -vvv --log-cli-level=info
                    --target-file       $<TARGET_FILE:tests_cpp_cuda_${FILE}>
                    --compile-commands  ${CMAKE_BINARY_DIR}/compile_commands.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    test_environment(tests_python_test_${FILE} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR}/python)

endfunction()

add_one_test(graph)
add_one_test(saxpy)

test_environment(tests_python_test_graph CMAKE_CUDA_COMPILER_ID set ${CMAKE_CUDA_COMPILER_ID})
