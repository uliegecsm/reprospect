# Fix for https://github.com/kokkos/kokkos-tools/issues/306.
set(KokkosTools_DIR $ENV{KokkosTools_ROOT}/lib/cmake)

find_package(
    KokkosTools
    REQUIRED
    COMPONENTS kp_nvtx_connector
)

add_one_example(allocation)
test_environment(examples_kokkos_view_allocation KOKKOS_TOOLS_NVTX_CONNECTOR_LIB set $<TARGET_FILE:KokkosTools::kp_nvtx_connector>)
