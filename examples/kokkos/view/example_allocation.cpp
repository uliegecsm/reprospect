#include "Kokkos_Core.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

#if !defined(KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC)
    #error "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is not defined."
#endif

/**
 * @file
 *
 * Companion of @ref examples/kokkos/view/example_allocation.py.
 */

namespace reprospect::examples::kokkos::view
{

template <typename MemorySpace> requires Kokkos::is_memory_space_v<MemorySpace>
class Allocation
{
public:
    using scalar_t = char;
    using view_t   = Kokkos::View<scalar_t*, MemorySpace>;

    /**
     * @name Sizes.
     *
     * When @c KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is defined, the behavior of allocating
     * a @c Kokkos::View in @c Kokkos::CudaSpace is dictated by a threshold of 40000 bytes on the allocation size, see
     * https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/Cuda/Kokkos_CudaSpace.cpp#L139.
     */
    ///@{
    static constexpr size_t size_below_threshold = 39000;
    static constexpr size_t size_above_threshold = 41000;
    ///@}

public:
    bool run_impl(const Kokkos::Cuda& exec, const size_t size) const
    {
        const Kokkos::Profiling::ScopedRegion outer(std::to_string(size));

        std::optional<view_t> data = std::nullopt;

        {
            /**
             * @c Kokkos::View allocation for @c Kokkos::CudaSpace happens in
             * https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/Cuda/Kokkos_CudaSpace.cpp#L194-L215.
             */
            const Kokkos::Profiling::ScopedRegion inner("allocation");
            data.emplace(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec), size);
        }

        {
            /**
             * @c Kokkos::View deallocation for @c Kokkos::CudaSpace happens in
             * https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/Cuda/Kokkos_CudaSpace.cpp#L355-L367.
             */
            const Kokkos::Profiling::ScopedRegion inner("deallocation");
            data.reset();
        }

        return !data.has_value();
    }

    bool run(const Kokkos::Cuda& exec) const
    {
        return this->run_impl(exec, size_below_threshold) &&
               this->run_impl(exec, size_above_threshold);
    }

protected:
    //! @note The constructor of @c Kokkos::Profiling::ScopedRegion cannot be called with a @c std::string_view.
    Kokkos::Profiling::ScopedRegion region {std::string(Kokkos::Impl::TypeInfo<MemorySpace>::name())};
};

} // namespace reprospect::examples::kokkos::view

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        cudaStream_t stream = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));

        {
            const Kokkos::Cuda exec{stream};

            Kokkos::Profiling::ProfilingSection profiling_section {"AllocationProfiling"};
            profiling_section.start();

            reprospect::examples::kokkos::view::Allocation<Kokkos::CudaSpace  >{}.run(exec);
            reprospect::examples::kokkos::view::Allocation<Kokkos::SharedSpace>{}.run(exec);

            profiling_section.stop();
        }

        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
    }
}
