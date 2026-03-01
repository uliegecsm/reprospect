#include "add.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/atomic/example_add_double256.py.
 */

namespace reprospect::examples::kokkos::atomic {
//! Similar to CUDA @c double4_32a.
struct alignas(4 * sizeof(double)) Double4Aligned32 {
    double x, y, z, w;

    friend __device__ constexpr Double4Aligned32 operator+(const Double4Aligned32& a, const Double4Aligned32& b) {
        return {.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w};
    }

    constexpr auto operator<=>(const Double4Aligned32&) const = default;
};
} // namespace reprospect::examples::kokkos::atomic

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard{argc, argv};
    {
        using scalar_t = ::reprospect::examples::kokkos::atomic::Double4Aligned32;

        static_assert(alignof(scalar_t) == 32);
        static_assert(sizeof(scalar_t) == 32);

        static_assert(std::is_trivially_copyable_v<scalar_t>);

        ::reprospect::examples::kokkos::atomic::AtomicAdd<scalar_t>::run(Kokkos::Cuda{}, scalar_t{1, 2, 3, 4});
    }
}
