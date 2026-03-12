#ifndef REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_NEWTONFRACTAL_HPP
#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_NEWTONFRACTAL_HPP

#include "Kokkos_Core.hpp"

namespace reprospect::examples::kokkos::complex {

//! Explicitly avoid @c Kokkos::pow.
template <std::floating_point RealType, Kokkos::MemorySpace Mem>
struct ZPow4MinOne {
    using complex_t = Kokkos::complex<RealType>;

    using roots_view_t = Kokkos::View<complex_t[4], Mem>;

    roots_view_t roots;

    ZPow4MinOne() = default;

    template <Kokkos::ExecutionSpace Exec>
    ZPow4MinOne(const Exec& exec)
        : roots(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "roots of z^4-1")) {
        std::array<complex_t, 4> roots_h{
            complex_t{ RealType(1),  RealType(0)},
            complex_t{-RealType(1),  RealType(0)},
            complex_t{ RealType(0),  RealType(1)},
            complex_t{ RealType(0), -RealType(1)}
        };
        Kokkos::deep_copy(exec, roots, typename roots_view_t::host_mirror_type{roots_h.data()});
    }

    KOKKOS_FUNCTION
    static constexpr auto value_and_derivative(const Kokkos::complex<RealType>& z)
        -> Kokkos::pair<Kokkos::complex<RealType>, Kokkos::complex<RealType>> {
        const auto z3 = z * z * z;
        return {z3 * z - RealType(1), RealType(4) * z3};
    }
};

//! Assume that the domain is [-1, 1] x [-1, 1].
template <std::floating_point RealType, Kokkos::MemorySpace Mem, typename Divisor, typename Function>
struct ComputeColors {
   public:
    using count_t = unsigned int;
    using complex_t = Kokkos::complex<RealType>;

    using count_view_t = Kokkos::View<count_t**, Mem>;

   public:
    template <Kokkos::ExecutionSpace Exec>
    ComputeColors(
        const Exec& exec,
        Function function_,
        const count_t width,
        const count_t height,
        const count_t max_iters_)
        : function(std::move(function_))
        , colors(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "colors"), width, height)
        , iterations(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "iterations"), width, height)
        , max_iters(max_iters_) {
    }

    template <Kokkos::ExecutionSpace Exec>
    void apply(const Exec& exec) const {
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(exec, {0, 0}, {colors.extent(0), colors.extent(1)}), *this);
    }

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T iwidth, const T iheight) const noexcept {
        complex_t z{-RealType(1) + iwidth * dwidth, -RealType(1) + iheight * dheight};
        for (count_t iter = 0; iter < max_iters; ++iter) {
            const auto [value, derivative] = function.value_and_derivative(z);
            z -= Divisor{}(value, derivative);
            for (count_t iroot = 0; iroot < function.roots.size(); ++iroot) {
                const auto diff = z - function.roots(iroot);
                if (Kokkos::fabs(diff.real()) < tolerance && Kokkos::fabs(diff.imag()) < tolerance) {
                    colors(iwidth, iheight) = iroot;
                    iterations(iwidth, iheight) = iter;
                    return;
                }
            }
        }
    }

   public:
    Function function;
    count_view_t colors;     //! The color is set to be the index of the closest root found.
    count_view_t iterations; //! How many iterations needed to converge.
    count_t max_iters;       //! Allowed iterations.

   private:
    RealType dwidth = RealType(2) / colors.extent(0);  //! Step along the x axis, assuming length of 2.
    RealType dheight = RealType(2) / colors.extent(1); //! Step along the y axis, assuming length of 2.
    static constexpr RealType tolerance = 1e-12;       //! Tolerance to achieve.
};

} // namespace reprospect::examples::kokkos::complex

#endif // REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_NEWTONFRACTAL_HPP
