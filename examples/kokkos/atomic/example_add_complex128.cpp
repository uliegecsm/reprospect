#include "add.hpp"

#if !defined(KOKKOS_ENABLE_COMPLEX_ALIGN)
    #error "KOKKOS_ENABLE_COMPLEX_ALIGN is not defined."
#endif

/**
 * @file
 *
 * Companion of @ref examples/kokkos/atomic/example_add_complex128.py.
 */

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        using scalar_t = Kokkos::complex<double>;

        /// @name Native atomic CAS is supported for 128-bit word if properly aligned and trivially copyable.
        ///       See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomiccas.
        ///@{
        static_assert( sizeof(scalar_t) == 16);
        static_assert(alignof(scalar_t) == 16);

        static_assert(std::is_trivially_copyable_v<scalar_t>);
        ///@}

        ::reprospect::examples::kokkos::atomic::AtomicAdd<scalar_t>::run(Kokkos::Cuda{}, scalar_t{1, 2});
    }
}
