# This file contains setup w.r.t. compiler flags related to warnings.
include_guard(GLOBAL)

# Compiler flags that we always want.
set(REPROSPECT_COMPILE_WARNINGS -Wall -Wextra -Werror)

foreach(flag IN LISTS REPROSPECT_COMPILE_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${flag}>)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
    elseif(CMAKE_CUDA_COMPILER_ID STREQUAL Clang)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:${flag}>)
    else()
        message(FATAL_ERROR "Unsupported CUDA compiler ID ${CMAKE_CUDA_COMPILER_ID}.")
    endif()
endforeach()

if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL Clang)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-error=unknown-cuda-version>)
endif()
