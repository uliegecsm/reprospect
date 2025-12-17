function(add_one_kokkos_example FILE)
    add_one_example(${FILE}
        EXTRA_PRIVATE_LINK_LIBS Kokkos::kokkoscore
        EXTRA_TEST_ENV KOKKOS_TOOLS_NVTX_CONNECTOR_LIB set $<TARGET_FILE:KokkosTools::kp_nvtx_connector>
    )

endfunction()

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
