function(add_one_example FILE)

    add_executable(examples_kokkos_${FILE})

    target_sources(examples_kokkos_${FILE} PRIVATE example_${FILE}.cpp)

    set_source_files_properties(example_${FILE}.cpp PROPERTIES LANGUAGE CUDA)

    target_link_libraries(examples_kokkos_${FILE} PRIVATE Kokkos::kokkoscore)

endfunction()

find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

add_one_example(integration)
