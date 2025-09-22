#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda-helpers/errors/cuda_runtime.hpp"

/**
 * @file
 *
 * Inspired by https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/.
 */

using index_t = decltype(dim3::x);

__global__
void saxpy(const index_t size, const float factor, const float* __restrict__ const vec_x, float* __restrict__ const vec_y)
{
    const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size)
        vec_y[index] += factor * vec_x[index];
}

int main()
{
    constexpr index_t size = 1024;

    cudaStream_t stream = nullptr;
    CUDA_HELPERS_CHECK_CUDART_CALL(cudaStreamCreate(&stream));

    float* vec_x = nullptr;
    float* vec_y = nullptr;

    CUDA_HELPERS_CHECK_CUDART_CALL(cudaMallocAsync(&vec_x, size * sizeof(float), stream)); 
    CUDA_HELPERS_CHECK_CUDART_CALL(cudaMallocAsync(&vec_y, size * sizeof(float), stream));

    {
        const std::vector<float> vec_x_h(size, 1.f), vec_y_h(size, 2.f);
        CUDA_HELPERS_CHECK_CUDART_CALL(cudaMemcpyAsync(vec_x, vec_x_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_HELPERS_CHECK_CUDART_CALL(cudaMemcpyAsync(vec_y, vec_y_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_HELPERS_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));
    }

    constexpr index_t block_size = 128;
    constexpr index_t grid_size  = (size + block_size - 1) / block_size;

    saxpy<<<dim3{grid_size, 1, 1}, dim3{block_size, 1, 1} , 0, stream>>>(size, 2.f, vec_x, vec_y);

    std::vector<float> result(size);
    CUDA_HELPERS_CHECK_CUDART_CALL(cudaMemcpyAsync(result.data(), vec_y, size * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CUDA_HELPERS_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));

    CUDA_HELPERS_CHECK_CUDART_CALL(cudaFreeAsync(vec_x, stream));
    CUDA_HELPERS_CHECK_CUDART_CALL(cudaFreeAsync(vec_y, stream));

    CUDA_HELPERS_CHECK_CUDART_CALL(cudaStreamDestroy(stream));

    for(const auto& elm : result) {
        if(elm != 4.f) throw std::runtime_error("wrong value");
    }

    return 0;
}
