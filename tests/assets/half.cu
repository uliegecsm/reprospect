#include "cuda_fp16.h"

//! Implement the power of 2 of @c __half (1 element per thread).
__global__ void
    pow2_individual(__half* __restrict__ dst, const __half* __restrict__ const src, const unsigned int size) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        const __half val = src[index];
        dst[index] = __hmul(val, val);
    }
}

//! Packed implementation by processing 2 elements per thread using @c __half2.
__global__ void pow2_packed(__half* __restrict__ dst, const __half* __restrict__ const src, const unsigned int size) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto index_2 = 2 * index;

    if (index_2 + 1 < size) {
        //! Load 2 consecutive @c __half values as a single @c __half2 (packed load).
        const __half2* const src_h2 = reinterpret_cast<const __half2*>(src);
        __half2* const dst_h2 = reinterpret_cast<__half2*>(dst);

        const __half2 val = src_h2[index];

        dst_h2[index] = __hmul2(val, val);
    } else if (index_2 < size) {
        //! Handle last element if size is odd.
        const __half val = src[index_2];
        dst[index_2] = __hmul(val, val);
    }
}
