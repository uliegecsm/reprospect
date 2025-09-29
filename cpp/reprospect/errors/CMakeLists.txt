target_sources(
    reprospect
    INTERFACE
    FILE_SET headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_SOURCE_DIR}/cpp/reprospect
        FILES
            cuda.hpp
            cuda_runtime.hpp
)
