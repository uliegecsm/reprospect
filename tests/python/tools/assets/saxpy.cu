#include "cuda.h"

//! Kernel that performs a @c saxpy.
__global__ void saxpy_kernel(const float cst_a, const float* __restrict__ const d_x, float* __restrict__ const d_y, const unsigned int size)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_y[idx] = cst_a * d_x[idx] + d_y[idx];
    }
}
