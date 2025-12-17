function(add_one_cuda_example FILE)
    add_one_example(${FILE}
        EXTRA_PRIVATE_LINK_LIBS nvtx3::nvtx3-cpp
    )
endfunction()

add_subdirectory(virtual_functions)
