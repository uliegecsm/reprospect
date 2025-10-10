# Define a fixture for GPU resource acquisition.
set(GPU_LOCK_DIR "/ci-locks")


# Define a test fixture that acquires the resource.
add_test(NAME reprospect_resource_acquisition
         COMMAND ${CMAKE_COMMAND} -E env
                 GPU_LOCK_FILE=${GPU_LOCK_FILE}
                 reprospect-resource-acquisition --resource= --acquire)
set_tests_properties(gpu_acquire PROPERTIES FIXTURES_SETUP gpu)

# Define a test fixture that releases the resource.
add_test(NAME gpu_release
         COMMAND ${CMAKE_COMMAND} -E echo "GPU released")
set_tests_properties(gpu_release PROPERTIES FIXTURES_CLEANUP gpu)

# AcquireGPU.cmake
file(WRITE "${ENV{GPU_LOCK_FILE}}" "")
file(OPEN "${ENV{GPU_LOCK_FILE}}" GPU_LOCK FILE_APPEND)
file(LOCK GPU_LOCK EXCLUSIVE)
message(STATUS "GPU lock acquired")
# Lock will be released when file is closed at end of test
