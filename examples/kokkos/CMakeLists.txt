function(add_one_kokkos_example FILE)
    add_one_example(${FILE}
        EXTRA_PRIVATE_LINK_LIBS Kokkos::kokkoscore
        EXTRA_TEST_ENV KOKKOS_TOOLS_NVTX_CONNECTOR_LIB set $<TARGET_FILE:KokkosTools::kp_nvtx_connector>
    )

    set(aoe_EXECUTABLE "${aoe_EXECUTABLE}" PARENT_SCOPE)

endfunction()

function(add_one_kokkos_example_benchmarking NAME)
    add_one_kokkos_example(${NAME})

    if(NOT DEFINED aoe_EXECUTABLE)
        message(FATAL_ERROR "aoe_EXECUTABLE must be defined.")
    endif()

    target_link_libraries(${aoe_EXECUTABLE} PRIVATE benchmark::benchmark)
    target_sources(${aoe_EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR}/examples/kokkos/benchmarking.main.cpp)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/examples/kokkos/benchmarking.main.cpp PROPERTIES LANGUAGE CUDA)
endfunction()

find_package(benchmark REQUIRED COMPONENTS benchmark)

find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

find_package(
    KokkosTools
    REQUIRED
    COMPONENTS kp_nvtx_connector
)

add_subdirectory(atomic)
add_subdirectory(complex)
add_subdirectory(graph)
add_subdirectory(half)
add_subdirectory(view)
