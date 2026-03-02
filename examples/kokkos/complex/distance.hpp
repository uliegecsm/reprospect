#ifndef DISTANCE_H
#define DISTANCE_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "examples/kokkos/complex/floatingpointtraits.hpp"


namespace internal
{

template<typename T>
T
epsilonForExponent(int exponent)
{
    using Traits = traits::FloatingPoint<T>;
    
    return exponent < Traits::smallestNormalNumberExponent ?
        
        // The epsilon is constant amongst the subnormal numbers.
        Traits::subnormalNumbersEpsilon :
        
        // The amount of numbers per exponent is constant amongst the
        // normal numbers and is equal to `2^bitsForSignificand`, with
        // `bitsForSignificand` being the number of bits used to define
        // the significand of a floating-point number.
        //
        // Knowing the amount of normal numbers that can be representated for
        // a given exponent `x`, the epsilon is equal to the absolute
        // difference `2^(x + 1) - 2^x` divided by the amount of numbers.
        //
        // After simplification, `(2^(x + 1) - 2^x) / 2^bitsForSignificand`
        // becomes `2^(x - bitsForSignificand)`.
        std::exp2(T(exponent - std::numeric_limits<T>::digits + 1));
}

template<typename T>
T
distanceTowardsExponentUpperBound(T value, int exponent)
{
    return (std::exp2(T(exponent + 1)) - value)
           / epsilonForExponent<T>(exponent);
}

template<typename T>
T
distanceTowardsExponentLowerBound(T value, int exponent)
{
    return (value - std::exp2(T(exponent)))
           / epsilonForExponent<T>(exponent);
}

template<typename T>
T
distance(T from, T to)
{
    using Traits = traits::FloatingPoint<T>;
    
    assert(std::isfinite(from));
    assert(std::isfinite(to));
    assert(from != to);
    assert(from < to);
    assert(from > 0);
    assert(to > 0);
    assert(std::signbit(from) == std::signbit(to));
    
    int exponentFrom = std::ilogb(from);
    int exponentTo = std::ilogb(to);
    T out = 0;
    
    if (exponentFrom == exponentTo) {
        out = (to - from) / epsilonForExponent<T>(exponentFrom);
    }
    else {
        // 2^x     2^(x+1)       2^(x+2)                   2^(x+3)
        //  |   |   |      |      |            |            |     
        //     from                            to                 
        //                                                        
        //           \___________/                                
        //                 ^                                      
        //          distance between                              
        //       `2^(exponentFrom + 1)`                           
        //         and `2^exponentTo`                             
        
        // Compute the distance between `2^(exponentFrom + 1)`
        // and `2^exponentTo`.
        
        assert(exponentFrom < exponentTo);
        
        if (exponentTo - exponentFrom > 1) {
            // Do every in-between exponents represent normal numbers?
            if (exponentFrom + 1 >= Traits::smallestNormalNumberExponent) {
                out += Traits::amountOfNormalNumbersPerExponent
                       * (exponentTo - exponentFrom - 1);
            }
            // Do every in-between exponents represent subnormal numbers?
            else if (exponentTo <= Traits::smallestNormalNumberExponent) {
                int exponent = exponentFrom + 1;
                while (exponent < exponentTo) {
                    out += std::exp2(
                        T(exponent - Traits::smallestSubnormalNumberExponent)
                    );
                    
                    ++exponent;
                }
            }
            else {
                int exponent = exponentFrom + 1;
                
                // Distance within subnormal numbers.
                while (exponent < Traits::smallestNormalNumberExponent) {
                    out += std::exp2(
                        T(exponent - Traits::smallestSubnormalNumberExponent)
                    );
                    
                    ++exponent;
                }
                
                // Distance within normal numbers.
                out += Traits::amountOfNormalNumbersPerExponent
                       * (exponentTo - exponent);
            }
        }
        
        // 2^x     2^(x+1)       2^(x+2)                   2^(x+3)
        //  |   |   |      |      |            |            |     
        //     from                            to                 
        //                                                        
        //       \_/               \__________/                   
        //        ^                     ^                         
        // distance towards      distance towards                 
        //     exponent              exponent                     
        //   upper bound           lower bound                    
        
        // Add both values at once to reduce the precision error
        // for large numbers.
        out += (
            distanceTowardsExponentUpperBound(from, exponentFrom)
            + distanceTowardsExponentLowerBound(to, exponentTo)
        );
    }
    
    assert(out >= 0);
    assert(out == std::trunc(out));
    return out;
}

} // namespace internal.


template<typename T>
T
distance(T from, T to)
{
    // The distance in ulps between two numbers x and y is equal to
    // the number of times that `std::nextafter()` needs to be called
    // in order to go from x to y.
    // The result is to become inaccurate if the difference between the
    // two numbers is of at least `2^std::numeric_limits<T>::digits`.
    
    if (std::isnan(from) || std::isnan(to)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    
    if (from == to) {
        return 0;
    }
    
    if (from > to) {
        std::swap(from, to);
    }
    
    if (std::isinf(from)) {
        // `from` equals to -infinity since `from < to`.
        return 1 + distance(-std::numeric_limits<T>::max(), to);
    }
    
    if (std::isinf(to)) {
        // `to` equals to +infinity since `from < to`.
        return 1 + distance(from, std::numeric_limits<T>::max());
    }
    
    if (std::signbit(from) && std::signbit(to)) {
        // `from` and `to` are negative.
        T positiveFrom = -from;
        from = -to;
        to = positiveFrom;
    }
    
    static constexpr T smallestSubnormal = std::numeric_limits<T>::denorm_min();
    
    // If either number to compare is null, the exponent returned by
    // `std::logb()` will be infinity. Hence we need to compute the exponent
    // of the nearest value from zero of the same sign then add one to
    // the result to make up for the distance from zero to the nearest value.
    
    if (from == 0) {
        // `to` has to be positive since `from < to` and `from == 0`.
        if (to == smallestSubnormal) {
            return 1;
        }
        
        return 1 + internal::distance(smallestSubnormal, to);
    }
    
    if (to == 0) {
        // `from` has to be negative since `from < to` and `to == 0`.
        if (from == -smallestSubnormal) {
            return 1;
        }
        
        return 1 + internal::distance(smallestSubnormal, -from);
    }
    
    // When the signs of the values to compare differ, we need to compare
    // the distance between each number with the nearest value from zero
    // of the same sign, then sum the results with an extra increment of
    // two to make up for the distances from zero to the nearest values.
    
    if (std::signbit(from) != std::signbit(to)) {
        // `from` has to be negative and `to` positive since `from < to`
        // and their sign differ.
        return 2
            + (from == -smallestSubnormal ? 0 :
                  internal::distance(smallestSubnormal, -from))
            + (to == smallestSubnormal ? 0 :
                internal::distance(smallestSubnormal, to));
    }
    
    return internal::distance(from, to);
}

#endif // DISTANCE_H