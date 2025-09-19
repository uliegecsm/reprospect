# This file contains setup w.r.t. compiler flags related to warnings.
include_guard(GLOBAL)

# Host compiler flags that we always want.
set(CUDA_HELPERS_COMPILE_WARNINGS -Wall -Wextra -Werror)

foreach(flag IN LISTS CUDA_HELPERS_COMPILE_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${flag}>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
