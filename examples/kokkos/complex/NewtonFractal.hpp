#ifndef REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_NEWTONFRACTAL_HPP
#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_NEWTONFRACTAL_HPP

#include "Kokkos_Core.hpp"

namespace reprospect::examples::kokkos::complex {

//! Explicitly avoid @c Kokkos::pow.
template <typename Mem, std::floating_point RealType = double> requires Kokkos::is_memory_space_v<Mem>
struct ZPow4MinOne {
    using complex_t = Kokkos::complex<RealType>;

    using roots_view_t = Kokkos::View<complex_t[4], Mem>;

    roots_view_t roots;

    ZPow4MinOne() = default;

    template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
    ZPow4MinOne(const Exec& exec) : roots(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "roots of z^4-1"))
    {
        std::array<complex_t, 4> roots_h{
            complex_t{1., 0.},
            complex_t{-1., 0.},
            complex_t{0., 1.},
            complex_t{0., -1. }
        };
        Kokkos::deep_copy(exec, roots, roots_view_t{roots_h.data()});
    }

    KOKKOS_FUNCTION
    static constexpr auto value_and_derivative(const Kokkos::complex<RealType>& z) -> Kokkos::pair<Kokkos::complex<RealType>, Kokkos::complex<RealType>> {
        const auto tmp = z * z * z;
        return {tmp * z - 1., 4. * tmp};
    }
};

//! Assume that the domain is [-1, -1] x [-1, 1].
template <typename Mem, std::floating_point RealType, typename Divisor, typename Function> requires Kokkos::is_memory_space_v<Mem>
struct ComputeColors {
    using count_t = unsigned int;
    using complex_t = Kokkos::complex<RealType>;

    using count_view_t = Kokkos::View<count_t**, Mem>;

    Function function;
    count_view_t colors; //! The color is set to be the index of the closest root found, see @ref roots.
    count_view_t iterations; //! How many iterations needed to converge.
    count_t max_iters; //! Allowed iterations.

    double dwidth = 2. / colors.extent(0); //! Step along the x axis, assuming length of 2.
    double dheight = 2. / colors.extent(1); //! Step along the y axis, assuming length of 2.
    double tolerance = 1e-12; //! Tolerance to achieve.

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T iwidth, const T iheight) const noexcept {
        complex_t z{-1. + iwidth * dwidth, -1. + iheight * dheight};
        for(count_t iter = 0; iter < max_iters; ++iter) {
            const auto [value, derivative] = function.value_and_derivative(z);
            z -= Divisor{}(value, derivative);
            for(count_t iroot = 0; iroot < function.roots.size(); ++iroot) {
                const complex_t diff = z - function.roots(iroot);
                if(Kokkos::abs(diff.real()) < tolerance && Kokkos::abs(diff.imag()) < tolerance) {
                    colors(iwidth, iheight) = iroot;
                    iterations(iwidth, iheight) = iter;
                    return;
                }
            }
        }
    }
};

} // namespace reprospect::examples::kokkos::complex

#endif // REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_NEWTONFRACTAL_HPP
