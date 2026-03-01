#include "Kokkos_Core.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

#include "examples/kokkos/complex/example_division.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/complex/example_division.py.
 */

namespace reprospect::examples::kokkos::complex {

class Division {
   public:
    static constexpr size_t size = 1024;

    using complex_view_t = Kokkos::View<Kokkos::complex<double>[size], Kokkos::CudaSpace>;

   public:
    Division()
        : src_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "src A"))
        , src_b(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "src B"))
        , dst_c(Kokkos::view_alloc(exec, "dst C")) {
    }

    void scaling_branch() const {
        reset();
        {
            const Kokkos::Profiling::ScopedRegion region("scaling_branch");
            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec, 0, size),
                Scaling<true, complex_view_t>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec.fence();
    }

    void scaling() const {
        reset();
        {
            const Kokkos::Profiling::ScopedRegion region("scaling");
            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec, 0, size),
                Scaling<false, complex_view_t>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec.fence();
    }

    void iec559() const {
        reset();
        {
            const Kokkos::Profiling::ScopedRegion region("iec559");
            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec, 0, size),
                Iec559<complex_view_t>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec.fence();
    }

    void reset() const {
        Kokkos::deep_copy(exec, src_a, {4, 2});
        Kokkos::deep_copy(exec, src_b, {2, 1});
    }

   private:
    Kokkos::Cuda exec{};
    complex_view_t src_a, src_b, dst_c;
};

} // namespace reprospect::examples::kokkos::complex

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard{argc, argv};
    {
        using namespace reprospect::examples::kokkos::complex;

        Kokkos::Profiling::ProfilingSection profiling_section{"division"};
        profiling_section.start();

        Division division{};

        division.scaling_branch();
        division.scaling();
        division.iec559();

        profiling_section.stop();
    }
}
