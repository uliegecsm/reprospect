add_test(
    NAME tests_tools_device_properties
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_device_properties.py -s --log-cli-level=info
)
test_environment(tests_tools_device_properties PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR})
