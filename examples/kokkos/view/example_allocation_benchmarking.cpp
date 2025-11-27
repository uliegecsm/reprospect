#include "Kokkos_Core.hpp"

#include "benchmark/benchmark.h"

#if !defined(KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC)
    #error "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is not defined."
#endif

/**
 * @file
 *
 *  Companion of @ref examples/kokkos/view/example_allocation_benchmarking.py.
 */

//! These partial overrides are only needed by 'nvcc' (as of @c Cuda 12.8.0).
#if defined(__NVCC__)
    //! Use this macro when only one @c __what__ method is overridden.
    #define FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(__what__) \
        void __what__(::benchmark::State& state) override { this->__what__(static_cast<const ::benchmark::State&>(state)); }
#else
    #define FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(...)
#endif

namespace reprospect::examples::kokkos::view
{
//! Type contained in allocated buffers.
using scalar_t = char;

struct MemoryPoolState
{
    uint64_t reserved_mem_current = 0;
    uint64_t reserved_mem_high = 0;
    uint64_t used_mem_current = 0;
    uint64_t used_mem_high = 0;
    uint64_t release_threshold = 0;
};

struct MemoryPool
{
    cudaMemPool_t ptr = nullptr;

    //! Get default memory pool of the device to which @p exec is associated.
    static MemoryPool get(const Kokkos::Cuda& exec)
    {
        cudaMemPool_t ptr = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaDeviceGetDefaultMemPool(&ptr, exec.cuda_device()));
        return MemoryPool{.ptr = ptr};
    }

    void trim(const size_t keep = 0) const {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolTrimTo(ptr, keep));
    }

    //! See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#resource-usage-statistics.
    void reset_watermarks() const
    {
        uint64_t value = 0;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolSetAttribute(ptr, cudaMemPoolAttrReservedMemHigh, &value));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolSetAttribute(ptr, cudaMemPoolAttrUsedMemHigh,     &value));
    }

    /**
     * Query attributes.
     *
     * References:
     *  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/#physical-page-caching-behavior
     */
    MemoryPoolState state() const
    {
        MemoryPoolState state {};

        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolGetAttribute(ptr, cudaMemPoolAttrReservedMemCurrent, &state.reserved_mem_current));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolGetAttribute(ptr, cudaMemPoolAttrReservedMemHigh,    &state.reserved_mem_high));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolGetAttribute(ptr, cudaMemPoolAttrUsedMemCurrent,     &state.used_mem_current));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolGetAttribute(ptr, cudaMemPoolAttrUsedMemHigh,        &state.used_mem_high));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPoolGetAttribute(ptr, cudaMemPoolAttrReleaseThreshold,   &state.release_threshold));

        return state;
    }
};

class Common : public ::benchmark::Fixture
{
public:
    static constexpr unsigned short int range_count = 0;
    static constexpr unsigned short int range_size  = 1;

public:
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(SetUp)
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(TearDown)

    void SetUp(const ::benchmark::State&) override
    {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        exec = Kokkos::Cuda(stream);

        //! Ensure that the memory pool is set at the same state for all tests.
        const auto memory_pool = MemoryPool::get(*exec);
        memory_pool.trim(0);
        memory_pool.reset_watermarks();
    }

    void TearDown(const ::benchmark::State&) override
    {
        exec.reset();
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
    }

private:
    cudaStream_t stream = nullptr;

protected:
    std::optional<Kokkos::Cuda> exec = std::nullopt;
};

//! Allocate using @c CUDA API.
template <bool Async>
class WithCUDA : public Common
{
public:
    using buffer_t = scalar_t*;

public:
    //! Allocate with @c cudaMallocAsync and synchronize the @c CUDA stream underlying @p exec.
    void allocate(buffer_t* const ptr, const int64_t size) const requires Async
    {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMallocAsync(
            (void**)ptr,
            size * sizeof(scalar_t),
            exec->cuda_stream()
        ));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(exec->cuda_stream()));
    }

    //! Allocate with @c cudaMalloc.
    void allocate(buffer_t* const ptr, const size_t size) const requires (!Async) {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void**)ptr, size * sizeof(scalar_t)));
    }

    //! Deallocate with @c cudaFreeAsync and synchronize stream using @c Cuda.
    void deallocate(buffer_t const ptr) const requires Async
    {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeAsync((void*)ptr, exec->cuda_stream()));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(exec->cuda_stream()));
    }

    //! Deallocate with @c cudaFree.
    void deallocate(buffer_t const ptr) const requires (!Async) {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree((void*)ptr));
    }

    auto run(const ::benchmark::State& state) const
    {
        std::vector<buffer_t> buffers(state.range(range_count), nullptr);

        for(auto& buffer : buffers)
            allocate(&buffer, state.range(range_size));

        for(const auto buffer : buffers)
            deallocate(buffer);
    }
};

//! Allocate using @c Kokkos.
template <bool Async>
class WithKokkos : public Common
{
public:
    using view_t = Kokkos::View<scalar_t*, Kokkos::Cuda>;

public:
    /**
     * @brief Pass an execution space instance to the @c Kokkos::View allocation.
     *
     * Ultimately, it ends up doing a @c cudaMallocAsync followed by a stream synchronization when the buffer size
     * is above the threshold.
     *
     * See https://github.com/kokkos/kokkos/blob/146241cf3a68454527994a46ac473861c2b5d4f1/core/src/Cuda/Kokkos_CudaSpace.cpp#L209-L220.
     */
    auto allocate(const size_t size) const requires Async {
        return view_t{Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec), size};
    }

    /**
     * @brief Do not pass an execution space instance to the @c Kokkos::View allocation.
     *
     * Ultimately, it ends up doing a @c cudaMallocAsync followed by a device synchronization when the buffer size
     * is above the threshold.
     *
     * See https://github.com/kokkos/kokkos/blob/146241cf3a68454527994a46ac473861c2b5d4f1/core/src/Cuda/Kokkos_CudaSpace.cpp#L209-L220.
     */
    auto allocate(const size_t size) const requires (!Async) {
        return view_t{Kokkos::view_alloc(Kokkos::WithoutInitializing), size};
    }

    auto run(const ::benchmark::State& state) const
    {
        std::vector<view_t> views(state.range(range_count));

        for(auto& view : views)
            view = allocate(state.range(range_size));

        views.clear();
    }
};

void parameters(::benchmark::internal::Benchmark* benchmark)
{
    static constexpr size_t threshold = 40000;

    benchmark
        ->ArgNames({"count", "size"})
        ->Args({ 1, threshold - 1000})->Args({ 1, threshold + 1000})
        ->Args({ 4, threshold - 1000})->Args({ 4, threshold + 1000})
        ->Args({ 8, threshold - 1000})->Args({ 8, threshold + 1000})
        ->Args({12, threshold - 1000})->Args({12, threshold + 1000})
        ->ArgsProduct({
            {4, 8, 12},
            ::benchmark::CreateRange(128, 128<<17, 4)
        })
        ->ArgsProduct({
            {1},
            ::benchmark::CreateRange(128, 128<<19, 4)
        });
}

#define REPROSPECT_EXAMPLES_KOKKOS_VIEW_ALLOCATION(_class_, _name_, _async_)            \
    BENCHMARK_TEMPLATE_DEFINE_F(_class_, _name_, _async_)(benchmark::State& state) {    \
        for(auto _ : state) this->run(state);                                           \
        const auto mps = MemoryPool::get(*exec).state();                                \
        state.counters["cudaMemPoolAttrReservedMemCurrent"] = mps.reserved_mem_current; \
        state.counters["cudaMemPoolAttrReservedMemHigh"   ] = mps.reserved_mem_high;    \
        state.counters["cudaMemPoolAttrUsedMemCurrent"    ] = mps.used_mem_current;     \
        state.counters["cudaMemPoolAttrUsedMemHigh"       ] = mps.used_mem_high;        \
        state.counters["cudaMemPoolAttrReleaseThreshold"  ] = mps.release_threshold;    \
    }                                                                                   \
    BENCHMARK_REGISTER_F(_class_, _name_)->Apply(parameters);

REPROSPECT_EXAMPLES_KOKKOS_VIEW_ALLOCATION(WithCUDA,   cuda_async,   true)
REPROSPECT_EXAMPLES_KOKKOS_VIEW_ALLOCATION(WithCUDA,   cuda,         false)
REPROSPECT_EXAMPLES_KOKKOS_VIEW_ALLOCATION(WithKokkos, kokkos_async, true)
REPROSPECT_EXAMPLES_KOKKOS_VIEW_ALLOCATION(WithKokkos, kokkos,       false)

} // namespace HELM::benchmarks::kokkos::core
