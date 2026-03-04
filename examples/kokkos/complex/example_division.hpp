#ifndef REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP
#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP

#include "Kokkos_Core.hpp"

namespace reprospect::examples::kokkos::complex {

/**
 * Adapted from https://github.com/jtravs/cuda_complex/blob/master/cuda_complex.hpp#L553.
 * Similar to https://github.com/NVIDIA/cccl/blob/a91db6e2a022a7aa03b37873f0d4caf5ac81281d/libcudacxx/include/cuda/std/__complex/complex.h#L400-L496.
 *
 * @todo Use @c Kokkos::scalbn and @c Kokkos::ilogb when available.
 */
template <bool EnableBranching, typename RealType>
KOKKOS_FUNCTION constexpr Kokkos::complex<RealType>
    logb_scalbn(const Kokkos::complex<RealType>& x, const Kokkos::complex<RealType>& y) {
    int ilogbw = 0;
    RealType a = x.real();
    RealType b = x.imag();
    RealType c = y.real();
    RealType d = y.imag();
    RealType logbw = Kokkos::logb(Kokkos::fmax(Kokkos::fabs(c), Kokkos::fabs(d)));
    if (Kokkos::isfinite(logbw)) {
        ilogbw = static_cast<int>(logbw);
        c = scalbn(c, -ilogbw);
        d = scalbn(d, -ilogbw);
    }
    RealType denom = c * c + d * d;
    RealType real = scalbn((a * c + b * d) / denom, -ilogbw);
    RealType imag = scalbn((b * c - a * d) / denom, -ilogbw);
    if constexpr (EnableBranching) {
        if (Kokkos::isnan(real) && Kokkos::isnan(imag)) {
            if ((denom == RealType(0)) && (!Kokkos::isnan(a) || !Kokkos::isnan(b))) {
                real = Kokkos::copysign(Kokkos::Experimental::infinity_v<RealType>, c) * a;
                imag = Kokkos::copysign(Kokkos::Experimental::infinity_v<RealType>, c) * b;
            } else if ((Kokkos::isinf(a) || Kokkos::isinf(b)) && Kokkos::isfinite(c) && Kokkos::isfinite(d)) {
                a = Kokkos::copysign(Kokkos::isinf(a) ? RealType(1) : RealType(0), a);
                b = Kokkos::copysign(Kokkos::isinf(b) ? RealType(1) : RealType(0), b);
                real = Kokkos::Experimental::infinity_v<RealType> * (a * c + b * d);
                imag = Kokkos::Experimental::infinity_v<RealType> * (b * c - a * d);
            } else if (
                Kokkos::isinf(logbw) && logbw > RealType(0) && Kokkos::isfinite(x.real())
                && Kokkos::isfinite(x.imag())) {
                c = Kokkos::copysign(Kokkos::isinf(c) ? RealType(1) : RealType(0), c);
                d = Kokkos::copysign(Kokkos::isinf(d) ? RealType(1) : RealType(0), d);
                real = RealType(0) * (a * c + b * d);
                imag = RealType(0) * (b * c - a * d);
            }
        }
    }
    return {real, imag};
}

//! Adaptor for @ref logb_scalbn.
template <bool EnableBranching>
struct DivisorLogbScalbn {
    template <typename... Args>
    KOKKOS_FUNCTION constexpr auto operator()(Args&&... args) const {
        return logb_scalbn<EnableBranching>(std::forward<Args>(args)...);
    }
};

} // namespace reprospect::examples::kokkos::complex

#endif // REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP
