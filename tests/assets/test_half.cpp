#include <array>

#include "cuda/runtime_helper.hpp"
#include "nvtx3/nvtx3.hpp"

#include "test_half.cu"

struct MyAppDomain {
    static constexpr char const * name{"half"};
};

//! Copy results back and check.
template <typename T, size_t S>
void check(
    const T* __restrict__ const dst,
    std::array<T, S>& buf,
    const std::array<T, S>& src,
    const cudaStream_t stream) {
    REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(buf.data(), dst, S * sizeof(T), cudaMemcpyDeviceToHost, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));
    for (unsigned int index = 0; index < S; ++index)
        if (buf[index] != src[index] * src[index])
            throw std::runtime_error("wrong value");
}

int main() {
    //! Use an odd size on purpose.
    constexpr unsigned int size = 129;
    constexpr size_t bytes = size * sizeof(__half);

    //! Stream.
    cudaStream_t stream = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamCreate(&stream));

    //! Host buffers.
    std::array<__half, size> src_h{};
    std::array<__half, size> dst_h{};

    // Initialize input
    for (unsigned int index = 0; index < size; ++index) {
        const float value(index % 10);
        src_h[index] = __float2half(value);
    }

    //! Device buffers.
    __half* src_d = nullptr;
    __half* dst_d = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&src_d, bytes, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&dst_d, bytes, stream));

    //! Copy to device.
    REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(src_d, src_h.data(), bytes, cudaMemcpyHostToDevice, stream));

    //! Launch the individual implementation.
    {
        const ::nvtx3::scoped_range_in<MyAppDomain> range{"individual"};
        constexpr dim3 block{size, 1, 1};
        pow2_individual<<<1, block, 0, stream>>>(dst_d, src_d, size);
        REPROSPECT_CHECK_CUDART_CALL(cudaGetLastError());
    }

    //! Copy results back and check.
    check(dst_d, dst_h, src_h, stream);

    //! Launch the packed version.
    {
        const ::nvtx3::scoped_range_in<MyAppDomain> range{"packed"};
        constexpr dim3 block{(size + 2 - 1) / 2, 1, 1};
        pow2_packed<<<1, block, 0, stream>>>(dst_d, src_d, size);
        REPROSPECT_CHECK_CUDART_CALL(cudaGetLastError());
    }

    //! Copy results back and check.
    check(dst_d, dst_h, src_h, stream);

    //! Cleanup.
    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(src_d, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(dst_d, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamDestroy(stream));

    return 0;
}
