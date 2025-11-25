# Detect GPUs.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/reprospect/utils/detect.py --sep=\; --cc
    OUTPUT_VARIABLE ReProspect_DETECTED_GPUS
    COMMAND_ERROR_IS_FATAL ANY
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(ReProspect_DETECTED_GPUS)
    message(STATUS "${CMAKE_PROJECT_NAME} detected the following GPU compute capability: ${ReProspect_DETECTED_GPUS}.")
else()
    message(STATUS "${CMAKE_PROJECT_NAME} did not detect any GPU.")
endif()

foreach(value IN LISTS ReProspect_DETECTED_GPUS)
    if(NOT value MATCHES "^[0-9]+$")
        message(FATAL_ERROR "Invalid detected compute capability value '${value}'.")
    endif()
endforeach()

# For now, we only support CMAKE_CUDA_ARCHITECTURES that contain exactly 1 value.
list(LENGTH CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_ARCHITECTURES_LENGTH)
if(NOT CMAKE_CUDA_ARCHITECTURES_LENGTH EQUAL 1)
    message(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires that CMAKE_CUDA_ARCHITECTURES(=${CMAKE_CUDA_ARCHITECTURES}) contains exactly 1 architecture.")
endif()

# The CMAKE_CUDA_ARCHITECTURES might be suffixed with "-real" or "-virtual",
# see also https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html.
# We strip any such suffix to ease checking inclusion in the detected GPU architectures.
set(ReProspect_ENABLED_CUDA_ARCHITECTURE "${CMAKE_CUDA_ARCHITECTURES}")
list(TRANSFORM ReProspect_ENABLED_CUDA_ARCHITECTURE REPLACE "-real$|-virtual$" "")
