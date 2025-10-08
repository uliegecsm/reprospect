function(add_one_example FILE)

    set(FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/example_${FILE}.cpp)

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILENAME})

    set(EXECUTABLE ${gprr_PARENT_RELPATH_REPR}_${FILE})

    add_executable(${EXECUTABLE})

    target_sources(${EXECUTABLE} PRIVATE ${FILENAME})

    set_source_files_properties(${FILENAME} PROPERTIES LANGUAGE CUDA)

    target_link_libraries(${EXECUTABLE} PRIVATE Kokkos::kokkoscore)

    add_test(NAME ${EXECUTABLE} COMMAND $<TARGET_FILE:${EXECUTABLE}>)

endfunction()

find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

add_one_example(integration)
