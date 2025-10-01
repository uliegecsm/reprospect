# This file contains setup w.r.t. compiler flags related to warnings.
include_guard(GLOBAL)

# Host compiler flags that we always want.
set(REPROSPECT_COMPILE_WARNINGS -Wall -Wextra -Werror)

foreach(flag IN LISTS REPROSPECT_COMPILE_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${flag}>)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
    endif()
endforeach()

if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
endif()
