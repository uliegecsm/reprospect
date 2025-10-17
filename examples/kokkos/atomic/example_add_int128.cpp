#include "add.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/atomic/example_add_int128.py.
 */

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        using scalar_t = __int128_t;

        static_assert( sizeof(scalar_t) == 16);
        static_assert(alignof(scalar_t) == 16);

        ::reprospect::examples::kokkos::atomic::AtomicAdd<scalar_t>::run(Kokkos::Cuda{}, scalar_t{42});
    }
}
