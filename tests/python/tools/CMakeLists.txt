function(add_pytest FILE)

    add_test(
        NAME tests_tools_${FILE}
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_${FILE}.py -s --log-cli-level=info
    )

    test_environment(tests_tools_${FILE} PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR}/python)
    test_environment(tests_tools_${FILE} CMAKE_BINARY_DIR set ${CMAKE_BINARY_DIR})
    test_environment(tests_tools_${FILE} CMAKE_CUDA_ARCHITECTURES set ${CMAKE_CUDA_ARCHITECTURES})

endfunction()

add_pytest(architecture)
add_pytest(binaries)
add_pytest(device_properties)
