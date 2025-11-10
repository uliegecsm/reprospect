add_test(
    NAME tests_nvtx_nvtx
    COMMAND nsys profile
                --trace=nvtx
                --sample=none
                --backtrace=none
                --cpuctxsw=none
                ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_nvtx.py
                    --config-file=${CMAKE_SOURCE_DIR}/tests/python/.pytest.ini
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
