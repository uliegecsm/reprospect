function(add_pytest FILE)

    add_test(
        NAME tests_tools_${FILE}
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_${FILE}.py -vvv -s --log-cli-level=info
    )

    test_environment(tests_tools_${FILE} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR}/python)
    test_environment(tests_tools_${FILE} CMAKE_BINARY_DIR set ${CMAKE_BINARY_DIR})
    test_environment(tests_tools_${FILE} CMAKE_CUDA_ARCHITECTURES set ${CMAKE_CUDA_ARCHITECTURES})

endfunction()

add_pytest(architecture)
add_pytest(binaries)
add_pytest(device_properties)

test_environment(tests_tools_binaries CMAKE_CUDA_COMPILER set ${CMAKE_CUDA_COMPILER})
test_environment(tests_tools_binaries CMAKE_CUDA_COMPILER_ID set ${CMAKE_CUDA_COMPILER_ID})
test_environment(tests_tools_binaries CMAKE_CUDA_COMPILER_LAUNCHER set ${CMAKE_CUDA_COMPILER_LAUNCHER})
test_environment(tests_tools_binaries CMAKE_CURRENT_BINARY_DIR set ${CMAKE_CURRENT_BINARY_DIR})
