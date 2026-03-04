#ifndef FLOATINGPOINTTRAITS_H
#define FLOATINGPOINTTRAITS_H

#include <limits>


namespace traits {

namespace internal {

template <typename T_Scalar>
constexpr T_Scalar powerOfTwo(T_Scalar exponent) {
    return exponent == 0 ? 1 : exponent > 0 ? (2 * powerOfTwo(exponent - 1)) : (1 / (2 * powerOfTwo(-exponent - 1)));
}

} // namespace internal.


template <typename T>
struct FloatingPoint {
    // The floating-point numbers are treated here as being defined in the form
    // `value = significand * 2^exponent` with the significand value ranging
    // between 1 and `std::numeric_limits<T>::radix` (usually 2).
    //
    // As a result, the exponents for normalized floating-point numbers are
    // defined between `std::numeric_limits<T>::min_exponent - 1` and
    // `std::numeric_limits<T>::max_exponent - 1`.

    static constexpr int smallestSubnormalNumberExponent = std::numeric_limits<T>::min_exponent
                                                         - std::numeric_limits<T>::digits;

    static constexpr int smallestNormalNumberExponent = std::numeric_limits<T>::min_exponent - 1;

    static constexpr int largestNumberExponent = std::numeric_limits<T>::max_exponent - 1;

    // Within the subnormal numbers range, incrementing the exponent
    // doubles the amount of numbers that can be represented within
    // two successive power of two but the epsilon value
    // between two adjacent numbers remains constant.
    static constexpr T subnormalNumbersEpsilon = std::numeric_limits<T>::denorm_min();

    // Within the normal numbers range, incrementing the exponent
    // halves the precision of the epsilon value between two adjacent numbers
    // but the amount of numbers that can be represented
    // within two successive power of two remains constant.
    static constexpr T amountOfNormalNumbersPerExponent = internal::powerOfTwo(T(std::numeric_limits<T>::digits - 1));
};

} // namespace traits.

#endif // FLOATINGPOINTTRAITS_H