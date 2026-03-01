#ifndef REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP
#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP

namespace reprospect::examples::kokkos::complex {
//! Adapted from https://github.com/jtravs/cuda_complex/blob/master/cuda_complex.hpp#L553.
template <typename T>
KOKKOS_FUNCTION Kokkos::complex<T> iec559(const Kokkos::complex<T>& __z, const Kokkos::complex<T>& __w) {
    int __ilogbw = 0;
    T __a = __z.real();
    T __b = __z.imag();
    T __c = __w.real();
    T __d = __w.imag();
    T __logbw = Kokkos::logb(Kokkos::fmax(Kokkos::fabs(__c), Kokkos::fabs(__d)));
    if (Kokkos::isfinite(__logbw)) {
        __ilogbw = static_cast<int>(__logbw);
        __c = scalbn(__c, -__ilogbw);
        __d = scalbn(__d, -__ilogbw);
    }
    T __denom = __c * __c + __d * __d;
    T __x = scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    T __y = scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (Kokkos::isnan(__x) && Kokkos::isnan(__y)) {
        if ((__denom == T(0)) && (!Kokkos::isnan(__a) || !Kokkos::isnan(__b))) {
            __x = Kokkos::copysign(T(INFINITY), __c) * __a;
            __y = Kokkos::copysign(T(INFINITY), __c) * __b;
        } else if ((Kokkos::isinf(__a) || Kokkos::isinf(__b)) && Kokkos::isfinite(__c) && Kokkos::isfinite(__d)) {
            __a = Kokkos::copysign(Kokkos::isinf(__a) ? T(1) : T(0), __a);
            __b = Kokkos::copysign(Kokkos::isinf(__b) ? T(1) : T(0), __b);
            __x = T(INFINITY) * (__a * __c + __b * __d);
            __y = T(INFINITY) * (__b * __c - __a * __d);
        } else if (Kokkos::isinf(__logbw) && __logbw > T(0) && Kokkos::isfinite(__a) && Kokkos::isfinite(__b)) {
            __c = Kokkos::copysign(Kokkos::isinf(__c) ? T(1) : T(0), __c);
            __d = Kokkos::copysign(Kokkos::isinf(__d) ? T(1) : T(0), __d);
            __x = T(0) * (__a * __c + __b * __d);
            __y = T(0) * (__b * __c - __a * __d);
        }
    }
    return {__x, __y};
}

template <typename RealType>
KOKKOS_FUNCTION constexpr RealType norm(const Kokkos::complex<RealType>& value) {
    return value.real() * value.real() + value.imag() * value.imag();
}

template <bool EnableBranching, std::floating_point RealType>
KOKKOS_FUNCTION constexpr Kokkos::complex<RealType>
    scaling(const Kokkos::complex<RealType>& x, const Kokkos::complex<RealType>& y) noexcept {
    const RealType scale = Kokkos::fmax(Kokkos::fabs(y.real()), Kokkos::fabs(y.imag()));
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

template <typename ViewType>
struct Iec559 {
    typename ViewType::const_type src_a, src_b;
    ViewType dst_c;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T idx) const {
        dst_c(idx) = iec559(src_a(idx), src_b(idx));
    }
};

template <bool EnableBranching, typename ViewType>
struct Scaling {
    ViewType::const_type src_a, src_b;
    ViewType dst_c;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const noexcept {
        dst_c(index) = scaling<EnableBranching>(src_a(index), src_b(index));
    }
};

} // namespace reprospect::examples::kokkos::complex

#endif // REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_EXAMPLE_DIVISION_HPP
