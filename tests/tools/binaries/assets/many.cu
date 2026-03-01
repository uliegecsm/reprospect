#include <cstdio>

//! Atomically add 42.
__device__ void atomic_add_42(float* __restrict__ const a, const unsigned int idx) {
    atomicAdd(&a[idx], 42.f);
}

//! Add two @c float.
__device__ float add(const float a, const float b) {
    return a + b;
}

//! Basic parallel vector addition (with offset of 42) with bounds checking.
__global__ void vector_atomic_add_42(
    const float* __restrict__ const a,
    const float* __restrict__ const b,
    float* __restrict__ const c,
    const unsigned int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = add(a[idx], b[idx]);
        atomic_add_42(c, idx);
    }
}

//! Say hi.
__global__ void say_hi() {
    printf("Hi from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}
