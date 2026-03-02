#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "examples/kokkos/complex/example_division.hpp"

namespace reprospect::examples::kokkos::complex {

using kcomplex_t = Kokkos::complex<double>;
using scomplex_t = std::complex<double>;

constexpr double inf = std::numeric_limits<double>::infinity();

// Matcher that handles NaN, signed inf, and approximate equality.
MATCHER_P(ComplexEq, expected,
           "complex value near (" + std::to_string(expected.real()) + ", " +
               std::to_string(expected.imag()) + ")") {
  auto match_scalar = [&](const double got, const double exp) -> bool {
    if (std::isnan(exp)) return std::isnan(got);
    if (std::isinf(exp)) return std::isinf(got) && (got > 0) == (exp > 0);
    return got == exp;
  };
  return match_scalar(arg.real(), expected.real()) &&
         match_scalar(arg.imag(), expected.imag());
}

//! If @ref expt_std is @c std::nullopt, results for @c std::complex are expected to agree with @ref expt_iec559.
struct DivisionCase {
    std::string_view label;
    kcomplex_t num;
    kcomplex_t den;
    kcomplex_t expt_iec559;
    std::optional<kcomplex_t> expt_std = std::nullopt;
};

class ComplexDivisionTest : public ::testing::TestWithParam<DivisionCase> {};

TEST_P(ComplexDivisionTest, Iec559) {
  const auto& p = GetParam();
  const kcomplex_t result = reprospect::examples::kokkos::complex::iec559(p.num, p.den);
  EXPECT_THAT(result, ComplexEq(p.expt_iec559)) << p.label;
}

TEST_P(ComplexDivisionTest, StdComplex) {
  const auto& p = GetParam();
  const scomplex_t num{p.num.real(), p.num.imag()};
  const scomplex_t den{p.den.real(), p.den.imag()};
  const scomplex_t result = num / den;
  const kcomplex_t expected = p.expt_std.value_or(p.expt_iec559);
  EXPECT_THAT(result, ComplexEq(expected)) << p.label;
}

INSTANTIATE_TEST_SUITE_P(
    AllCases, ComplexDivisionTest,
    ::testing::Values(
        // ── Both methods agree ────────────────────────────────────────────────
        DivisionCase{"pure_real_by_pure_real",
                {6., 0.}, {2., 0.}, {3., 0.}},

        DivisionCase{"pure_real_by_pure_imag",
                {1., 0.}, {0., 2.}, {0., -0.5}},

        DivisionCase{"pure_imag_by_pure_real",
                {0., 4.}, {3., 0.}, {0., 4. / 3.}},

        DivisionCase{"all_negative_parts",
                {-3., -7.}, {-1., -2.}, {3.4, 0.2}, std::nullopt},

        DivisionCase{"general",
                {1., 2.}, {3., 4.}, {0.44, 0.08}},

        DivisionCase{"both_large_same_magnitude",
                {1.e+154, 1.e+154}, {1.e+154, 1.e+154}, {1., 0.}, std::nullopt},

        DivisionCase{"baudin_fig6b_large_num",
                {0x1p1023, 0x1p1023}, {1., 1.}, {inf, 0.}},

        // ── Methods diverge ───────────────────────────────────────────────────
        // Denominator real part much larger than imaginary part:
        // naive c^2+d^2 flushes to 0, iec559 scales correctly.
        DivisionCase{"denom_real_dominates",
                {1., 1.}, {1.e+200, 1.e-05},
                /*iec559*/ {1.e-200, 1.e-200}},

        DivisionCase{"denom_imag_dominates",
                {1., 1.}, {1.e-05, 1.e+200},
                /*iec559*/ {1e-200, -1e-200}},

        // c^2 + d^2 overflows in naive implementation.
        DivisionCase{"denom_c2_d2_overflow",
                {1., 0.}, {1.e+154, 1.e+154},
                /*iec559*/ {5.e-155, -5.e-155}},

        // Numerator near DBL_MAX: naive multiply overflows.
        DivisionCase{"num_real_near_dbl_max",
                {1.7976931348623157e+308, 0.}, {2., 0.},
                /*iec559*/ {8.988465674311579e+307, 0.}},

        // Baudin 2.3: large imaginary denominator.
        DivisionCase{"baudin_2_3",
                {1., 1.}, {1., 0x1p1023},
                /*iec559*/ {0x1p-1023, -0x1p-1023}},

        // Baudin 2.5: both parts large but different scales.
        DivisionCase{"baudin_2_5",
                {0x1p1023, 0x1p-1023}, {0x1p677, 0x1p-677},
                /*iec559*/ {0x1p346, 0.},
                /*std*/    kcomplex_t{0x1p346, -0x1p-1008}},

        // Baudin fig 6a: subnormal inputs — naive produces NaN.
        DivisionCase{"baudin_fig6a",
                {0x1p-1074, 0x1p-1074}, {0x1p-1073, 0x1p-1074},
                /*iec559*/ {0.6, 0.2}},

        DivisionCase{"baudin_fig6a_",
                {0x1p1023, 0x1p1023}, {1., 1.},
                /*iec559*/ {inf, 0.}}
    ),
    [](const ::testing::TestParamInfo<DivisionCase>& info) {
      return std::string(info.param.label);
    });

} // namespace reprospect::examples::kokkos::complex