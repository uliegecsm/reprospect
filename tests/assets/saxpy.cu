#include "cuda.h"

using index_t = decltype(dim3::x);

//! Kernel that performs a @c saxpy.
__global__ void saxpy_kernel(
    const index_t size,
    const float cst_a,
    const float* __restrict__ const d_x,
    float* __restrict__ const d_y) {
    const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_y[idx] = cst_a * d_x[idx] + d_y[idx];
    }
}
