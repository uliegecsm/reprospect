target_sources(
    cuda_helpers
    INTERFACE
    FILE_SET headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_SOURCE_DIR}/cpp/cuda-helpers
        FILES
            cuda.hpp
            cuda_runtime.hpp
)
