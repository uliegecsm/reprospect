#include <functional>
#include <iostream>
#include <source_location>

#include "Kokkos_Core.hpp"

#include "examples/kokkos/complex/example_division.hpp"

#if !defined(__STDC_IEC_60559_COMPLEX__)
#    error "Your compiler does not comply to ISO/IEC 60559."
#endif

/**
 * @addtogroup unittests
 *
 * compliance of complex division w.r.t. ISO/IEC 60559
 * ---------------------------------------------------
 *
 * Tests to check compliance to ISO/IEC 60559 of several complex division implementations.
 *
 * See also @cite iso-iec-9899-2024, section G.5.2.
 *
 * The tests can be found in @ref examples/kokkos/complex/example_division_compliance.cpp.
 */

namespace reprospect::examples::kokkos::complex {

#define CHECK_STATEMENT(statement, computed)                                                                           \
    if (!check(#statement, (statement), (computed), std::source_location::current()))                                  \
        ++failures;

/**
 * According to @cite iso-iec-9899-2024, section G.3:
 *
 *  A complex or imaginary value with at least one infinite part is regarded as an infinity (even if its other part is a quiet NaN).
 */
#define IS_INF(value) Kokkos::isinf(value.real()) || Kokkos::isinf(value.imag())

template <template <typename> typename ComplexType, typename Divisor>
struct Compliance {
    using real_t = double;
    using complex_t = ComplexType<real_t>;

    static constexpr real_t pinf = Kokkos::Experimental::infinity_v<real_t>;
    static constexpr real_t minf = -pinf;
    static constexpr real_t zero = real_t(0);
    static constexpr real_t one = real_t(1);

    static bool check(
        const char* statement,
        const bool passed,
        const complex_t& computed,
        const std::source_location loc = std::source_location::current()) noexcept {
        if (!passed) {
            std::cerr << "[FAIL] " << loc.function_name() << '\n'
                      << "       " << statement << '\n'
                      << "       " << "but got " << computed << '\n'
                      << '(' << loc.file_name() << ':' << loc.line() << std::endl;
        }
        return passed;
    }

    /// If the first operand is an infinity and the second operand is a finite
    /// number, then the result of the @c operator/ is an infinity.
    static unsigned int test_infinite_divided_by_finite(unsigned int failures = 0) noexcept {
        const auto res_a = Divisor{}(complex_t{pinf, zero}, complex_t{one, one});
        CHECK_STATEMENT(IS_INF(res_a), res_a);

        const auto res_b = Divisor{}(complex_t{pinf, minf}, complex_t{3, 2});
        CHECK_STATEMENT(IS_INF(res_b), res_b);

        const auto res_c = Divisor{}(complex_t{minf, one}, complex_t{one, zero});
        CHECK_STATEMENT(IS_INF(res_c), res_c);

        const auto res_d = Divisor{}(complex_t{Kokkos::Experimental::quiet_NaN_v<real_t>, minf}, complex_t{one, zero});
        CHECK_STATEMENT(IS_INF(res_d), res_d)

        return failures;
    }

    /// If the first operand is a finite number and the second operand is an
    /// infinity, then the result of the @c operator/ is a zero.
    static unsigned int test_finite_divided_by_infinite(unsigned int failures = 0) noexcept {
        const auto res_a = Divisor{}(complex_t{one, one}, complex_t{pinf, zero});
        CHECK_STATEMENT(res_a.real() == zero, res_a);
        CHECK_STATEMENT(res_a.imag() == zero, res_a);

        const auto res_b = Divisor{}(complex_t{one, one}, complex_t{minf, pinf});
        CHECK_STATEMENT(res_b.real() == zero, res_b);
        CHECK_STATEMENT(res_b.imag() == zero, res_b);

        const auto res_c = Divisor{}(complex_t{one, one}, complex_t{zero, pinf});
        CHECK_STATEMENT(res_c.real() == zero, res_c);
        CHECK_STATEMENT(res_c.imag() == zero, res_c);

        return failures;
    }

    /// If the first operand is a nonzero finite number or an infinity and the
    /// second operand is a zero, then the result of the @c operator/ is an infinity.
    static unsigned int test_nonzero_divided_by_zero(unsigned int failures = 0) noexcept {
        const auto res_a = Divisor{}(complex_t{one, one}, complex_t{zero, zero});
        CHECK_STATEMENT(IS_INF(res_a), res_a);

        const auto res_b = Divisor{}(complex_t{pinf, zero}, complex_t{zero, zero});
        CHECK_STATEMENT(IS_INF(res_b), res_b);

        const auto res_c = Divisor{}(complex_t{one, zero}, complex_t{zero, -zero});
        CHECK_STATEMENT(IS_INF(res_c), res_c);

        return failures;
    }

    /**
     * Sanity checks:
     * * \f$ (1+i) / (1+i) = 1 \f$
     * * \f$ (2+2i) / (2+2i) = 1 \f$
     */
    static unsigned int test_sanity(unsigned int failures = 0) noexcept {
        const auto res_a = Divisor{}(complex_t{one, one}, complex_t{one, one});
        CHECK_STATEMENT((res_a == complex_t{one, zero}), res_a);

        const auto res_b = Divisor{}(complex_t{real_t(2), real_t(2)}, complex_t{real_t(2), real_t(2)});
        CHECK_STATEMENT((res_b == complex_t{one, zero}), res_b);

        return failures;
    }
};

} // namespace reprospect::examples::kokkos::complex

#define CHECK_COMPLEX_COMPLIANT_IMPL(_test_, type, divisor, expt, compare)                                             \
    std::cout << "=== " << #type << " with " << #divisor << " is " << expt << "compliant to " << #_test_               \
              << " ===" << std::endl;                                                                                  \
    if (reprospect::examples::kokkos::complex::Compliance<type, divisor>::test_##_test_() compare) {                   \
        throw std::runtime_error("Should be " expt "compliant.");                                                      \
    }

#define CHECK_COMPLEX_COMPLIANT(test, type, divisor)     CHECK_COMPLEX_COMPLIANT_IMPL(test, type, divisor, "", != 0)
#define CHECK_COMPLEX_NOT_COMPLIANT(test, type, divisor) CHECK_COMPLEX_COMPLIANT_IMPL(test, type, divisor, "NOT ", == 0)

int main() {
    using namespace reprospect::examples::kokkos::complex;

    using logb_scalbn_t = DivisorLogbScalbn<true, false>;
    using ilogb_scalbn_t = DivisorLogbScalbn<true, true>;

    CHECK_COMPLEX_COMPLIANT(sanity, std::complex, std::divides<void>)
    CHECK_COMPLEX_COMPLIANT(sanity, Kokkos::complex, logb_scalbn_t)
    CHECK_COMPLEX_COMPLIANT(sanity, Kokkos::complex, ilogb_scalbn_t)
    CHECK_COMPLEX_COMPLIANT(sanity, Kokkos::complex, std::divides<void>)

    CHECK_COMPLEX_COMPLIANT(infinite_divided_by_finite, std::complex, std::divides<void>)
    CHECK_COMPLEX_COMPLIANT(infinite_divided_by_finite, Kokkos::complex, logb_scalbn_t)
    CHECK_COMPLEX_COMPLIANT(infinite_divided_by_finite, Kokkos::complex, ilogb_scalbn_t)
    CHECK_COMPLEX_NOT_COMPLIANT(infinite_divided_by_finite, Kokkos::complex, std::divides<void>)

    CHECK_COMPLEX_COMPLIANT(finite_divided_by_infinite, std::complex, std::divides<void>)
    CHECK_COMPLEX_COMPLIANT(finite_divided_by_infinite, Kokkos::complex, logb_scalbn_t)
    CHECK_COMPLEX_COMPLIANT(finite_divided_by_infinite, Kokkos::complex, ilogb_scalbn_t)
    CHECK_COMPLEX_NOT_COMPLIANT(finite_divided_by_infinite, Kokkos::complex, std::divides<void>)

    CHECK_COMPLEX_COMPLIANT(nonzero_divided_by_zero, std::complex, std::divides<void>)
    CHECK_COMPLEX_COMPLIANT(nonzero_divided_by_zero, Kokkos::complex, logb_scalbn_t)
    CHECK_COMPLEX_COMPLIANT(nonzero_divided_by_zero, Kokkos::complex, ilogb_scalbn_t)
    CHECK_COMPLEX_COMPLIANT(nonzero_divided_by_zero, Kokkos::complex, std::divides<void>)

    return EXIT_SUCCESS;
}
