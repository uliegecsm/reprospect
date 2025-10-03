#include "Kokkos_Core.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

#if !defined(KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC)
    #error "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is not defined."
#endif

namespace reprospect::examples::kokkos::view
{

class Allocation
{
public:
    using scalar_t = char;
    using view_t   = Kokkos::View<scalar_t*, Kokkos::CudaSpace>;

public:
    Allocation() { profiling_section.start(); }
    ~Allocation() { profiling_section.stop(); }

    bool run(const Kokkos::Cuda& exec, const size_t size) const
    {
        const Kokkos::Profiling::ScopedRegion outer(std::to_string(size));

        std::optional<view_t> data = std::nullopt;

        {
            /**
             * @c Kokkos::View allocation for @c Kokkos::CudaSpace happens in
             * https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/Cuda/Kokkos_CudaSpace.cpp#L194-L215.
             */
            const Kokkos::Profiling::ScopedRegion region("allocation");
            data.emplace(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec), size);
        }

        {
            /**
             * @c Kokkos::View deallocation for @c Kokkos::CudaSpace happens in
             * https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/Cuda/Kokkos_CudaSpace.cpp#L355-L367.
             */
            const Kokkos::Profiling::ScopedRegion region("deallocation");
            data.reset();
        }

        return !data.has_value();
    }

protected:
    Kokkos::Profiling::ProfilingSection profiling_section {"Allocation"};
};

} // namespace reprospect::examples::kokkos::view

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        const reprospect::examples::kokkos::view::Allocation test {};

        cudaStream_t stream = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));

        {
            const Kokkos::Cuda exec{stream};

            /**
             * The behavior of allocating a @c Kokkos::View in @c Kokkos::CudaSpace when @c KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC
             * is dictated by a threshold of 40000 bytes on the allocation size, see
             * https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/Cuda/Kokkos_CudaSpace.cpp#L139.
             */
            test.run(exec, 39000);
            test.run(exec, 41000);
        }

        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
    }
}
