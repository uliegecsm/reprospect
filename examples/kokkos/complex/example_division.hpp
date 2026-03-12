#ifndef REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP
#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP

#include "Kokkos_Core.hpp"

namespace reprospect::examples::kokkos::complex {

/**
 * @c std::ilogb produces @c FP_ILOGBNAN for a NaN input. Yet, it is implementation-defined whether it
 * maps to @c INT_MIN or @c INT_MAX @cite cppreference-ilogb @cite iso-iec-9899-2024.
 *
 * However, @c std::ilogb outputs @c INT_MAX on infinite inputs. Thus, if @c FP_ILOGBNAN maps
 * to @c INT_MAX, there is no way to distinguish between a NaN or an infinite input.
 */
#if !defined(FP_ILOGBNAN) || !defined(INT_MIN)
#    error "Missing required macros FP_ILOGBNAN / INT_MIN."
#endif

#if FP_ILOGBNAN == INT_MIN
#    define ILOGB_NAN_INF_DISTINGUISHABLE 1
#endif

/**
 * Adapted from https://github.com/jtravs/cuda_complex/blob/master/cuda_complex.hpp#L553.
 * Similar to https://github.com/NVIDIA/cccl/blob/a91db6e2a022a7aa03b37873f0d4caf5ac81281d/libcudacxx/include/cuda/std/__complex/complex.h#L400-L496.
 *
 * @tparam EnableCompliance Enable ISO/IEC 60559 compliance.
 * @tparam UseIlogb         Use @c ilogb instead of @c logb.
 *                          Only available when @c ILOGB_NAN_INF_DISTINGUISHABLE is defined.
 */
template <bool EnableCompliance, bool UseIlogb, typename RealType>
KOKKOS_FUNCTION constexpr Kokkos::complex<RealType>
    logb_scalbn(const Kokkos::complex<RealType>& x, const Kokkos::complex<RealType>& y) {
#if !defined(ILOGB_NAN_INF_DISTINGUISHABLE)
    static_assert(!UseIlogb);
#endif

    RealType a = x.real();
    RealType b = x.imag();
    RealType c = y.real();
    RealType d = y.imag();

    int ilogbw = 0;
    [[maybe_unused]]
    int iexp;
    [[maybe_unused]]
    RealType logbw{};

    if constexpr (UseIlogb) {
        iexp = Kokkos::ilogb(Kokkos::fmax(Kokkos::fabs(c), Kokkos::fabs(d)));
        if (iexp != FP_ILOGB0 && iexp != FP_ILOGBNAN && iexp != INT_MAX) {
            ilogbw = iexp;
            c = Kokkos::scalbn(c, -ilogbw);
            d = Kokkos::scalbn(d, -ilogbw);
        }
    } else {
        logbw = Kokkos::logb(Kokkos::fmax(Kokkos::fabs(c), Kokkos::fabs(d)));
        if (Kokkos::isfinite(logbw)) {
            ilogbw = static_cast<int>(logbw);
            c = Kokkos::scalbn(c, -ilogbw);
            d = Kokkos::scalbn(d, -ilogbw);
        }
    }

    RealType denom = c * c + d * d;
    RealType real = Kokkos::scalbn((a * c + b * d) / denom, -ilogbw);
    RealType imag = Kokkos::scalbn((b * c - a * d) / denom, -ilogbw);
    if constexpr (EnableCompliance) {
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
                [&]() {
                    if constexpr (UseIlogb) {
                        return iexp == INT_MAX;
                    } else {
                        return Kokkos::isinf(logbw);
                    }
                }()
                && Kokkos::isfinite(x.real()) && Kokkos::isfinite(x.imag())) {
                c = Kokkos::copysign(Kokkos::isinf(c) ? RealType(1) : RealType(0), c);
                d = Kokkos::copysign(Kokkos::isinf(d) ? RealType(1) : RealType(0), d);
                real = RealType(0) * (a * c + b * d);
                imag = RealType(0) * (b * c - a * d);
            }
        }
    }
    return {real, imag};
}

/**
 * Adapted from https://github.com/kokkos/kokkos/blob/9174877d49528ab293b6c7f4c6bd932a429b200a/core/src/Kokkos_Complex.hpp#L182-L206.
 * Similar to https://github.com/NVIDIA/thrust/blob/756c5afc0750f1413da05bd2b6505180e84c53d4/thrust/detail/complex/arithmetic.h#L119.
 */
template <bool EnableBranching, typename RealType>
KOKKOS_FUNCTION constexpr Kokkos::complex<RealType>
    norm_division(const Kokkos::complex<RealType>& x, const Kokkos::complex<RealType>& y) noexcept {
    const RealType scale = Kokkos::fabs(y.real()) + Kokkos::fabs(y.imag());
    if constexpr (EnableBranching) {
        if (scale == RealType(0)) {
            return {x.real() / scale, x.imag() / scale};
        }
    }
    const Kokkos::complex<RealType> x_scaled = x / scale;
    const Kokkos::complex<RealType> y_conj_scaled = Kokkos::conj(y) / scale;
    const RealType y_conj_scaled_norm = norm(y_conj_scaled);
    return (x_scaled * y_conj_scaled) / y_conj_scaled_norm;
}

//! Adaptor for @ref logb_scalbn.
template <bool EnableCompliance, bool UseIlogb = false>
struct DivisorLogbScalbn {
    template <typename... Args>
    KOKKOS_FUNCTION constexpr auto operator()(Args&&... args) const {
        return logb_scalbn<EnableCompliance, UseIlogb>(std::forward<Args>(args)...);
    }
};

//! Adaptor for @ref norm_division.
template <bool EnableBranching>
struct DivisorNormDivision {
    template <typename... Args>
    KOKKOS_FUNCTION constexpr auto operator()(Args&&... args) const {
        return norm_division<EnableBranching>(std::forward<Args>(args)...);
    }
};

} // namespace reprospect::examples::kokkos::complex

#endif // REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP
