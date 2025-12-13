#include "cuda.h"

template <typename T>
struct alignas(2 * sizeof(T)) MyAlignedStruct
{
    T real;
    T imag;
};

//! Kernel that uses "wide" loads and stores (128 bytes).
__global__ void wide_load_store_kernel(MyAlignedStruct<double>* __restrict__ const dst, const MyAlignedStruct<double>* __restrict__ const src)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    dst[idx] = src[idx];
}
