#include "cuda.h"

//! Kernel that uses shared memory.
__global__ void shared_memory_kernel(float* const data, const unsigned int size) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr unsigned int max_size = 128;

    __shared__ float values[max_size];

    if (idx < size && idx < max_size) {
        values[idx] = data[idx];

        __syncthreads();

        data[idx] = (values[idx > 0 ? idx - 1 : max_size] + values[idx < max_size ? idx + 1 : 0]) * 0.5f;
    }
}
