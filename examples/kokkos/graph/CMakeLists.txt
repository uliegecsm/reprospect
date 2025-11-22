add_one_example(dispatch)
test_environment(examples_kokkos_graph_dispatch KOKKOS_TOOLS_NVTX_CONNECTOR_LIB set $<TARGET_FILE:KokkosTools::kp_nvtx_connector>)

add_one_example(then)
test_environment(examples_kokkos_graph_then KOKKOS_TOOLS_NVTX_CONNECTOR_LIB set $<TARGET_FILE:KokkosTools::kp_nvtx_connector>)
