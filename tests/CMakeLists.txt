add_subdirectory(cpp)
add_subdirectory(python)

add_test(
    NAME    tests_saxpy
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_saxpy.py
                --name           saxpy
                --file           $<TARGET_FILE:tests_cpp_cuda_saxpy>
                --cwd            ${CMAKE_CURRENT_BINARY_DIR}
)
test_environment(tests_saxpy PYTHONPATH path_list_prepend ${CMAKE_SOURCE_DIR}/python)