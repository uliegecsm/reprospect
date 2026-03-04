function(add_one_kokkos_example FILE)
    add_one_example(${FILE}
        EXTRA_PRIVATE_LINK_LIBS Kokkos::kokkoscore
        EXTRA_TEST_ENV KOKKOS_TOOLS_NVTX_CONNECTOR_LIB set $<TARGET_FILE:KokkosTools::kp_nvtx_connector>
    )
endfunction()

function(add_one_kokkos_example_benchmarking NAME)
    add_one_example(${NAME}_benchmarking EXTRA_PRIVATE_LINK_LIBS benchmark_common)
endfunction()

find_package(benchmark REQUIRED COMPONENTS benchmark)
find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

find_package(
    KokkosTools
    REQUIRED
    COMPONENTS kp_nvtx_connector
)

add_library(benchmark_common OBJECT)
target_sources(benchmark_common PRIVATE ${CMAKE_SOURCE_DIR}/examples/kokkos/benchmarking.main.cpp)
set_source_files_properties(${CMAKE_SOURCE_DIR}/examples/kokkos/benchmarking.main.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(benchmark_common PUBLIC benchmark::benchmark Kokkos::kokkoscore)

add_subdirectory(atomic)
add_subdirectory(complex)
add_subdirectory(graph)
add_subdirectory(half)
add_subdirectory(view)
