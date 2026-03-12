add_one_kokkos_example(alignment)
add_one_example(division_compliance STANDALONE EXTRA_PRIVATE_LINK_LIBS Kokkos::kokkoscore)
add_one_kokkos_example(division)
add_one_kokkos_example(division_plot)
add_one_kokkos_example_benchmarking(division)

# Ensure that transitive shared library dependencies that are never actually called
# (e.g. libcuda.so.1 pulled in by Kokkos) are not listed as NEEDED by the linker,
# so the executable can run on machines without the NVIDIA driver installed.
# See also https://bnikolic.co.uk/blog/gnu-ld-as-needed.html.
target_link_options(examples_kokkos_complex_division_compliance PRIVATE "LINKER:--as-needed")
