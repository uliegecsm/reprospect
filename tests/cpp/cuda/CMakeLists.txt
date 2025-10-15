function(add_one_test FILE)

    set(FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/test_${FILE}.cpp)

    get_parent_relpath_repr(DIRECTORY ${CMAKE_SOURCE_DIR} FILENAME ${FILENAME})

    set(EXECUTABLE ${gprr_PARENT_RELPATH_REPR}_${FILE})

    add_executable(${EXECUTABLE})

    target_sources(${EXECUTABLE} PRIVATE ${FILENAME})

    target_link_libraries(${EXECUTABLE} PRIVATE CUDA::cudart CUDA::nvtx3)

    set_source_files_properties(${FILENAME} PROPERTIES LANGUAGE CUDA)

    if(ReProspect_ENABLED_CUDA_ARCHITECTURE IN_LIST ReProspect_DETECTED_GPUS)
        add_test(NAME ${EXECUTABLE} COMMAND $<TARGET_FILE:${EXECUTABLE}>)
    else()
        message(WARNING "Not adding test '${EXECUTABLE}' because ${ReProspect_ENABLED_CUDA_ARCHITECTURE} is not in the detected GPUs (${ReProspect_DETECTED_GPUS}).")
    endif()
endfunction()

find_package(CUDAToolkit REQUIRED COMPONENTS cudart nvtx3)

add_one_test(graph)
add_one_test(saxpy)
