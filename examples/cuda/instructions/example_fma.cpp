#include <fstream>
#include <string_view>

#include "cuda/runtime_helper.hpp"
#include "nvtx3/nvtx3.hpp"

/**
 * @file
 *
 * Companion of @ref examples/cuda/instructions/example_fma.py.
 */

namespace reprospect::examples::cuda::instructions
{

constexpr float cst_a = 1.000001f;
constexpr float cst_b = 1.000002f;

template <typename IndexType, IndexType unrolling>
__device__ float work(const IndexType niters)
{
    float res = 1.f;

    #pragma unroll unrolling
    for(IndexType iter = 0; iter < niters; ++iter) {
        res = fmaf(res, cst_a, cst_b);
    }

    return res;
}

template <typename IndexType, IndexType unrolling>
__global__ __launch_bounds__(1, 1)
void single_thread_clock(long long* __restrict__ const elapsed, float* __restrict__ const result, const IndexType niters)
{
    const long long start = clock64();

    result[0] = work<IndexType, unrolling>(niters);

    const long long stop = clock64();

    *elapsed = stop - start;
}

template <typename IndexType, IndexType unrolling>
__global__ __launch_bounds__(1, 1)
void single_thread(float* __restrict__ const result, const IndexType niters) {
    result[0] = work<IndexType, unrolling>(niters);
}

template <typename IndexType, IndexType unrolling>
__global__
void many_threads(float* __restrict__ const result, const IndexType niters) {
    result[0] = work<IndexType, unrolling>(niters);
}

//! @todo "independent" is shitty naming.
template <typename IndexType, IndexType unrolling>
class FMA
{
public:
    using value_t = float;
    using clock_t = long long; //! See https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#clock-and-clock64.

public:
    explicit FMA(const cudaStream_t stream)
    {
        REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&result,  sizeof(value_t), stream));
        REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&elapsed, sizeof(clock_t), stream));
    }

    ~FMA() {
        REPROSPECT_CHECK_CUDART_CALL(cudaFree(result));
        REPROSPECT_CHECK_CUDART_CALL(cudaFree(elapsed));
    }

    //! Launch @ref single_thread_clock.
    void single_thread_clock(const cudaStream_t stream, const char* out) const
    {
        {
            const nvtx3::scoped_range range("single_thread_clock");
            ::reprospect::examples::cuda::instructions::single_thread_clock<IndexType, unrolling><<<1, 1, 0, stream>>>(elapsed, result, unrolling);
        }

        clock_t elapsed_h = 0;
        REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(&elapsed_h, elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost, stream));
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));

        std::ofstream file(out);
        if(!file)
            throw std::runtime_error("Failed to open file.");
        file << "niters: " << unrolling << std::endl;
        file << "cycles: " << elapsed_h << std::endl;
    }

    //! Launch @ref single_thread.
    void single_thread(const cudaStream_t stream) const
    {
        {
            const nvtx3::scoped_range range("single_thread");
            ::reprospect::examples::cuda::instructions::single_thread<IndexType, unrolling><<<1, 1, 0, stream>>>(result, unrolling);
        }
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));
    }

    //! Launch @ref many_threads.
    void many_threads(const cudaStream_t stream) const
    {
        int device = 0;
        REPROSPECT_CHECK_CUDART_CALL(cudaGetDevice(&device));

        int multi_processor_count = 0;
        REPROSPECT_CHECK_CUDART_CALL(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));

        //! The value 48 is intended to ensure there are enough eligible warps per smsp to hide latencies.
        dim3 grid{static_cast<unsigned int>(multi_processor_count) * 48, 1, 1};
        dim3 block{1, 128, 1};

        {
            const nvtx3::scoped_range range("many_threads");
            ::reprospect::examples::cuda::instructions::many_threads<IndexType, unrolling><<<grid, block, 0, stream>>>(result, unrolling);
        }
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));
    }

protected:
    value_t* result = nullptr;
    clock_t* elapsed;
};

} // namespace reprospect::examples::cuda::instructions

int main(int argc, char* argv[])
{
    using namespace reprospect::examples::cuda::instructions;

    if(argc < 2)
        throw std::runtime_error("Requires a command.");

    const std::string_view command(argv[1]);

    if(command == "clock") {
        if(argc != 3) {
            throw std::runtime_error("Requires a file path argument to dump the clock cycles.");
        }
    }
    else if(command == "no-clock") {
        if(argc != 2) {
            throw std::runtime_error("No argument required.");
        }
    } else {
        throw std::runtime_error("Unsupported command.");
    }

    cudaStream_t stream;
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamCreate(&stream));

    {
        FMA<unsigned int, 128<<4> fma{stream};

        [[maybe_unused]]const auto range = ::nvtx3::start_range("fma");

        if(command == "clock") {
            fma.single_thread_clock(stream, argv[2]);
        } else {
            fma.single_thread(stream);
            fma.many_threads(stream);
        }
    }

    REPROSPECT_CHECK_CUDART_CALL(cudaStreamDestroy(stream));

    return EXIT_SUCCESS;
}
