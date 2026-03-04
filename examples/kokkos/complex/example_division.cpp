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
    static constexpr size_t size = 128 << 1;

    using real_t = double;
    using complex_t = Kokkos::complex<real_t>;
    using complex_view_t = Kokkos::View<complex_t[size], Kokkos::CudaSpace>;

    /**
     * @brief Measure the raw cost of a division implementation, deliberately avoiding memory pressure.
     *
     * Operands are derived from the hardware clock counter @c clock64 so that the compiler
     * cannot constant-fold or eliminate any arithmetic performed by @c Divisor.
     * No memory load or store is issued; the single store to @c dst
     * is predicated on an unreachable index value @c Kokkos::Experimental::finite_max_v<T>
     * to prevent dead-code elimination while never executing in practice.
     */
    template <typename Divisor, unsigned short Repeat = 1>
    struct JustDivide {
        complex_view_t dst;

        static __device__ auto get_clock64() noexcept {
            return static_cast<real_t>(clock64());
        }

        template <std::integral T>
        __device__ void operator()(const T index) const noexcept {
            complex_t res = complex_t{1, 1};
            for (unsigned int i = 0; i < Repeat; ++i)
                res -= Divisor{}(complex_t{get_clock64(), get_clock64()}, complex_t{get_clock64(), get_clock64()});
            if (index == Kokkos::Experimental::finite_max_v<T>) {
                dst(0) = res;
            }
        }
    };

   public:
    Division(const Kokkos::Cuda& exec)
        : dst(Kokkos::view_alloc(exec, "dst")) {
    }

    template <typename Divisor>
    void run(const Kokkos::Cuda& exec, const std::string& label) const {
        {
            const Kokkos::Profiling::ScopedRegion region(label);
            Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, size), JustDivide<Divisor>{.dst = dst});
        }
        exec.fence();
    }

   private:
    complex_view_t dst;
};

} // namespace reprospect::examples::kokkos::complex

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard{argc, argv};
    {
        using namespace reprospect::examples::kokkos::complex;

        const Kokkos::Cuda exec{};

        Division division{exec};

        Kokkos::Profiling::ProfilingSection profiling_section{"division"};
        profiling_section.start();

        division.run<DivisorLogbScalbn<true, false>>(exec, "LogbScalbn");
        division.run<DivisorLogbScalbn<true, true>>(exec, "ILogbScalbn");
        division.run<DivisorScaling<true>>(exec, "Scaling");

        profiling_section.stop();
    }
}
