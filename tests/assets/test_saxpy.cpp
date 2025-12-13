#include <stdexcept>
#include <vector>

#include "common/cuda_runtime.hpp"
#include "cub/detail/nvtx3.hpp"

/**
 * @file
 *
 * Inspired by https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/.
 */

using index_t = decltype(dim3::x);

__global__
void saxpy_kernel(const index_t size, const float factor, const float* __restrict__ const vec_x, float* __restrict__ const vec_y)
{
    const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size)
        vec_y[index] += factor * vec_x[index];
}

struct MyAppDomain{ static constexpr char const* name {"application_domain"}; };

int main()
{
    //! Mark the start of the application.
    ::nvtx3::mark_in<MyAppDomain>("Starting my application.");

    //! This one is superfluous but serves the tests.
    const auto& outer = ::nvtx3::start_range_in<MyAppDomain>("outer_useless_range");

    constexpr index_t size = 1024;

    //! Create streams.
    cudaStream_t stream_A = nullptr, stream_B = nullptr;
    {
        const ::nvtx3::scoped_range_in<MyAppDomain> range{"create_streams"};
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamCreate(&stream_A));
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamCreate(&stream_B));
    }

    float* vec_x = nullptr;
    float* vec_y = nullptr;

    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&vec_x, size * sizeof(float), stream_A));
    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&vec_y, size * sizeof(float), stream_B));

    {
        const ::nvtx3::scoped_range_in<MyAppDomain> range{"initialize_data"};
        const std::vector<float> vec_x_h(size, 1.f), vec_y_h(size, 2.f);
        REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(vec_x, vec_x_h.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream_A));
        REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(vec_y, vec_y_h.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream_B));
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream_A));
    }

    constexpr index_t block_size = 128;
    constexpr index_t grid_size  = (size + block_size - 1) / block_size;

    {
        const ::nvtx3::scoped_range_in<MyAppDomain> range{"launch_saxpy_kernel_first_time"};
        saxpy_kernel<<<dim3{grid_size, 1, 1}, dim3{block_size, 1, 1} , 0, stream_B>>>(size, 2.f, vec_x, vec_y);
        REPROSPECT_CHECK_CUDART_CALL(cudaGetLastError());
    }
    {
        ::nvtx3::scoped_range_in<MyAppDomain> range{"launch_saxpy_kernel_second_time"};
        saxpy_kernel<<<dim3{grid_size, 1, 1}, dim3{block_size, 1, 1} , 0, stream_B>>>(size, 2.f, vec_x, vec_y);
        REPROSPECT_CHECK_CUDART_CALL(cudaGetLastError());
    }

    std::vector<float> result(size);
    REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(result.data(), vec_y, size * sizeof(float), cudaMemcpyDeviceToHost, stream_B));

    REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream_B));

    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(vec_x, stream_A));
    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(vec_y, stream_B));

    REPROSPECT_CHECK_CUDART_CALL(cudaStreamDestroy(stream_A));
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamDestroy(stream_B));

    for(const auto& elm : result) {
        if(elm != 6.f) throw std::runtime_error("wrong value");
    }

    ::nvtx3::end_range_in<MyAppDomain>(outer);

    return 0;
}
