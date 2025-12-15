#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/half/example_math.py.
 */

namespace reprospect::examples::kokkos::half
{
enum class Method : std::uint8_t
{
    CUDA_HMAX = 0,
    FMAX = 1,
    KOKKOS_FMAX = 2
};

template <Method method, typename ViewType>
struct FunctorMax
{
    ViewType dst;
    typename ViewType::const_type src_a, src_b;

    template <std::integral T> requires (method == Method::CUDA_HMAX)
    KOKKOS_FUNCTION void operator()(const T idx) const {
        dst(idx) = __hmax(src_a(idx).operator __half(), src_b(idx).operator __half());
    }

    template <std::integral T> requires (method == Method::FMAX)
    KOKKOS_FUNCTION void operator()(const T idx) const {
        dst(idx) = fmax(src_a(idx).operator float(), src_b(idx).operator float());
    }

    template <std::integral T> requires (method == Method::KOKKOS_FMAX)
    KOKKOS_FUNCTION void operator()(const T idx) const {
        dst(idx) = Kokkos::fmax(src_a(idx), src_b(idx));
    }
};

class Max
{
private:
    using view_t = Kokkos::View<Kokkos::Experimental::half_t*, Kokkos::Cuda>;

    static constexpr size_t size = 128;

public:
    static void run(const Kokkos::Cuda& exec)
    {
        const view_t dst  (Kokkos::view_alloc(exec), size);
        const view_t src_a(Kokkos::view_alloc(exec), size);
        const view_t src_b(Kokkos::view_alloc(exec), size);

        Kokkos::deep_copy(exec, src_a, 42);
        Kokkos::deep_copy(exec, src_b, 666);

        Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, size), FunctorMax<Method::CUDA_HMAX,   view_t>{.dst = dst, .src_a = src_a, .src_b = src_b});
        Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, size), FunctorMax<Method::FMAX,        view_t>{.dst = dst, .src_a = src_a, .src_b = src_b});
        Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, size), FunctorMax<Method::KOKKOS_FMAX, view_t>{.dst = dst, .src_a = src_a, .src_b = src_b});

        exec.fence();
    }
};
} // namespace reprospect::examples::kokkos::half

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        reprospect::examples::kokkos::half::Max::run(Kokkos::Cuda{});
    }
}
