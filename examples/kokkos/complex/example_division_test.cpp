#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "examples/kokkos/complex/distance.hpp"
#include "examples/kokkos/complex/example_division.hpp"

namespace reprospect::examples::kokkos::complex {

using kcomplex_t = Kokkos::complex<double>;
using scomplex_t = std::complex<double>;

constexpr double inf = std::numeric_limits<double>::infinity();

template <typename T>
std::string get_msg(const T& value) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(25);
    oss << "complex value near (" << value << ")";
    return oss.str();
}

// Matcher that handles NaN, signed inf, and approximate equality.
MATCHER_P(ComplexEq, expected, get_msg(expected)) {
    auto match_scalar = [&](const double got, const double exp) -> bool {
        if (std::isnan(exp))
            return std::isnan(got);
        if (std::isinf(exp))
            return std::isinf(got) && (got > 0) == (exp > 0);
        return got == exp;
    };
    std::cout << "Comparing " << arg << " with expected " << expected << std::endl;
    return match_scalar(arg.real(), expected.real()) && match_scalar(arg.imag(), expected.imag());
}

//! If @ref expt_std is @c std::nullopt, results for @c std::complex are expected to agree with @ref expt_iec559. Similar for @ref expt_scaling.
struct DivisionCase {
    std::string_view label;
    kcomplex_t num;
    kcomplex_t den;
    kcomplex_t expt_iec559;
    std::optional<kcomplex_t> expt_std = std::nullopt;
    std::optional<kcomplex_t> expt_scaling = std::nullopt;
};

class ComplexDivisionTest : public ::testing::TestWithParam<DivisionCase> { };

// like cuda cccl
TEST_P(ComplexDivisionTest, Iec559) {
    std::cout << std::scientific << std::setprecision(50);
    const auto& p = GetParam();
    const kcomplex_t result = reprospect::examples::kokkos::complex::iec559(p.num, p.den);
    EXPECT_THAT(result, ComplexEq(p.expt_iec559)) << p.label;
}

// like old kokkos
TEST_P(ComplexDivisionTest, Scaling) {
    std::cout << std::scientific << std::setprecision(50);
    const auto& p = GetParam();
    const kcomplex_t result = reprospect::examples::kokkos::complex::scaling<true>(p.num, p.den);
    EXPECT_THAT(result, ComplexEq(p.expt_scaling.value_or(p.expt_iec559))) << p.label;
}

TEST_P(ComplexDivisionTest, StdComplex) {
    std::cout << std::scientific << std::setprecision(50);
    const auto& p = GetParam();
    const scomplex_t num{p.num.real(), p.num.imag()};
    const scomplex_t den{p.den.real(), p.den.imag()};
    const scomplex_t result = num / den;
    const kcomplex_t expected = p.expt_std.value_or(p.expt_iec559);
    EXPECT_THAT(result, ComplexEq(expected)) << p.label;
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    ComplexDivisionTest,
    ::testing::Values(
        // All methods agree.
        DivisionCase{
            "pure_real_by_pure_real",
            {6., 0.},
            {2., 0.},
            {3., 0.}
},

        DivisionCase{"pure_real_by_pure_imag", {1., 0.}, {0., 2.}, {0., -0.5}},

        DivisionCase{"pure_imag_by_pure_real", {0., 4.}, {3., 0.}, {0., 4. / 3.}},

        DivisionCase{
            .label = "all_negative_parts",
            .num = {-3., -7.},
            .den = {-1., -2.},
            .expt_iec559 = kcomplex_t{3.4, 0.2},
            .expt_scaling = kcomplex_t{3.4, std::nextafter(std::nextafter(0.2, 1), 1)}},

        DivisionCase{
            .label = "general",
            .num = {1., 2.},
            .den = {3., 4.},
            .expt_iec559 = kcomplex_t{0.44, 0.08},
            .expt_scaling = kcomplex_t{std::nextafter(0.44, 1), std::nextafter(std::nextafter(0.08, -1), 1)}},

        DivisionCase{"both_large_same_magnitude", {1.e+154, 1.e+154}, {1.e+154, 1.e+154}, {1., 0.}},

        DivisionCase{"div_by_zero", {1., 1.}, {0., 0.}, {inf, inf}},
        DivisionCase{"div_by_zero_", {1., 0.}, {0., 0.}, {inf, std::nan("")}},
        DivisionCase{"div_by_zero__", {0., 1.}, {0., 0.}, {std::nan(""), inf}},
        DivisionCase{"div_by_zero___", {0., 0.}, {0., 0.}, {std::nan(""), std::nan("")}},

        // ── Methods diverge ───────────────────────────────────────────────────
        // Denominator real part much larger than imaginary part:
        // naive c^2+d^2 flushes to 0, iec559 scales correctly.
        DivisionCase{
            "denom_real_dominates",
            {1., 1.},
            {1.e+200, 1.e-05},
            /*iec559*/ {1.e-200, 1.e-200}},

        DivisionCase{
            "denom_imag_dominates",
            {1., 1.},
            {1.e-05, 1.e+200},
            /*iec559*/ {1e-200, -1e-200}},

        // c^2 + d^2 overflows in naive implementation.
        DivisionCase{
            "denom_c2_d2_overflow",
            {1., 0.},
            {1.e+154, 1.e+154},
            /*iec559*/ {5.e-155, -5.e-155}},

        // Numerator near DBL_MAX: naive multiply overflows.
        DivisionCase{
            "num_real_near_dbl_max",
            {1.7976931348623157e+308, 0.},
            {2., 0.},
            /*iec559*/ {8.988465674311579e+307, 0.}},

        // Baudin 2.3: large imaginary denominator.
        DivisionCase{
            "baudin_2_3",
            {1., 1.},
            {1., 0x1p1023},
            /*iec559*/ {0x1p-1023, -0x1p-1023}},

        // Baudin 2.5: both parts large but different scales.
        DivisionCase{
            .label = "baudin_2_5",
            .num = {0x1p1023, 0x1p-1023},
            .den = {0x1p677, 0x1p-677},
            .expt_iec559 = {0x1p346, 0.},
            .expt_std = kcomplex_t{0x1p346, -0x1p-1008}},

        /// Values in Fig. 6 from @cite baudin-2012-robust-complex-division-scilab.
        /// Both scaling and iec559 agree, but seems to have a higher failure rate.
        DivisionCase{"baudin_2012_fig6_1", {1., 1.}, {1., 0x1p1023}, {0x1p-1023, -0x1p-1023}},
        DivisionCase{
            .label = "baudin_2012_fig6_2",
            .num = {1., 1.},
            .den = {0x1p-1023, 0x1p-1023},
            .expt_iec559 = kcomplex_t{0x1p1023, 0.},
            .expt_std = kcomplex_t{0x1p1023, 0.}},
        DivisionCase{
            .label = "baudin_2012_fig6_3",
            .num = {0x1p1023, 0x1p-1023},
            .den = {0x1p677, 0x1p-677},
            .expt_iec559 = kcomplex_t{0x1p346, 0.},
            .expt_std = kcomplex_t{0x1p346, -0x1p-1008}},
        DivisionCase{
            .label = "baudin_fig6_4",
            .num = {0x1p1023, 0x1p1023},
            .den = {1., 1.},
            .expt_iec559 = kcomplex_t{inf, 0.},
            .expt_scaling = kcomplex_t{0x1p1023, 0.}},
        DivisionCase{
            .label = "baudin_fig6_5",
            .num = {0x1p1020, 0x1p-844},
            .den = {0x1p656, 0x1p-780},
            .expt_iec559 = kcomplex_t{0x1p364, 0.},
            .expt_std = kcomplex_t{0x1p364, -0x1p-1072}},
        DivisionCase{
            .label = "baudin_fig6_6",
            .num = {0x1p-71, 0x1p1021},
            .den = {0x1p1001, 0x1p-323},
            .expt_iec559 = kcomplex_t{0x1p-1072, 0x1p20}},
        DivisionCase{
            .label = "baudin_fig6_8",
            .num = {0x1p-1074, 0x1p-1074},
            .den = {0x1p-1073, 0x1p-1074},
            .expt_iec559 = kcomplex_t{0.5, 0.5},
            .expt_std = kcomplex_t{0.6, 0.2},
            .expt_scaling = kcomplex_t{0.6, std::nextafter(0.2, -1)}}),
    [](const ::testing::TestParamInfo<DivisionCase>& info) { return std::string(info.param.label); });

//! Generate a random double.
struct Random {
    std::mt19937_64 gen;
    std::uniform_int_distribution<int> exp_dist{-1022, 1023};
    std::uniform_int_distribution<uint64_t> mant_dist{0, (1ULL << 52) - 1};
    std::bernoulli_distribution sign_dist{0.5};

    Random(const uint64_t seed = 0)
        : gen(seed) {
    }

    double operator()() noexcept {
        int exp = exp_dist(gen);
        uint64_t mant = mant_dist(gen);
        bool sign = sign_dist(gen);

        uint64_t bits = (uint64_t(sign) << 63) | (uint64_t(exp + 1023) << 52) | mant;

        double x;
        std::memcpy(&x, &bits, sizeof(double));

        return x;
    }
};

//! Reference division using the naïve formula, in extended precision.
std::complex<long double> reference_div(const std::complex<long double>& a, const std::complex<long double>& b) {
    long double ar = a.real();
    long double ai = a.imag();
    long double br = b.real();
    long double bi = b.imag();

    long double denom = br * br + bi * bi;

    if (denom == 0.L)
        return {NAN, NAN};

    long double real = (ar * br + ai * bi) / denom;
    long double imag = (ai * br - ar * bi) / denom;

    return {real, imag};
}

//! Display a histogram.
void print_histogram(const std::string_view name, const std::map<uint64_t, uint64_t>& histogram) {
    uint64_t total = 0;
    long double weighted_sum = 0.;
    uint64_t max_ulp = 0;

    for (const auto& [ulp, count]: histogram) {
        total += count;
        weighted_sum += (long double) ulp * count;
        max_ulp = std::max(max_ulp, ulp);
    }

    long double mean = total ? weighted_sum / total : 0.;

    auto count_if_le = [&](uint64_t bound) {
        uint64_t c = 0;
        for (const auto& [ulp, count]: histogram)
            if (ulp <= bound)
                c += count;
        return c;
    };

    std::cout << "\n=================================================\n";
    std::cout << "Results for: " << name << "\n";
    std::cout << "-------------------------------------------------\n";
    std::cout << "Total samples      : " << total << "\n";
    std::cout << "Mean ULP           : " << std::setprecision(10) << mean << "\n";
    std::cout << "Max ULP            : " << max_ulp << "\n";
    std::cout << "Exact (0 ULP)      : " << 100. * count_if_le(0) / total << " %\n";
    std::cout << "≤ 1 ULP            : " << 100. * count_if_le(1) / total << " %\n";
    std::cout << "≤ 2 ULP            : " << 100. * count_if_le(2) / total << " %\n";
    std::cout << "≤ 4 ULP            : " << 100. * count_if_le(4) / total << " %\n";
    std::cout << "≤ 5 ULP            : " << 100. * count_if_le(5) / total << " %\n";
    std::cout << "≤ 6 ULP            : " << 100. * count_if_le(6) / total << " %\n";

    std::cout << "\nHistogram (first 15 buckets):\n";

    size_t printed = 0;
    for (const auto& [ulp, count]: histogram) {
        if (printed++ >= 15)
            break;

        std::cout << "ULP=" << std::setw(4) << ulp << " | " << std::setw(10) << count << " | " << std::fixed
                  << std::setprecision(4) << (100. * count / total) << " %\n";
    }

    if (histogram.size() > 15)
        std::cout << "... (truncated)\n";
}

TEST(ComplexDivision, failure_rate) {
    using complex_view_t = Kokkos::View<kcomplex_t*, Kokkos::HostSpace>;
    using rcomplex_t = std::complex<long double>;
    using reference_view_t = Kokkos::View<rcomplex_t*, Kokkos::HostSpace>;

    constexpr size_t size = 1000; // set this higher for better sampling

    Kokkos::ScopeGuard guard{};

    const complex_view_t src_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "src A"), size);
    const complex_view_t src_b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "src B"), size);

    const complex_view_t dst_iec559(Kokkos::view_alloc(Kokkos::WithoutInitializing, "iec559"), size);
    const complex_view_t dst_scaling(Kokkos::view_alloc(Kokkos::WithoutInitializing, "scaling"), size);
    const complex_view_t dst_std(Kokkos::view_alloc(Kokkos::WithoutInitializing, "std"), size);

    const reference_view_t reference(Kokkos::view_alloc(Kokkos::WithoutInitializing, "reference"), size);

    Random random(0);

    std::map<uint64_t, uint64_t> histogram_iec559{}, histogram_scaling{}, histogram_std{};

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, size), [&](const size_t index) {
        src_a(index) = kcomplex_t{random(), random()};
        src_b(index) = kcomplex_t{random(), random()};

        dst_iec559(index) = reprospect::examples::kokkos::complex::iec559(src_a(index), src_b(index));
        dst_scaling(index) = reprospect::examples::kokkos::complex::scaling<true>(src_a(index), src_b(index));
        dst_std(index) = src_a(index) / src_b(index);

        reference(index) = reference_div(
            rcomplex_t{src_a(index).real(), src_a(index).imag()}, rcomplex_t{src_b(index).real(), src_b(index).imag()});

        const auto ulp_iec559 = std::max(
            distance(dst_iec559(index).real(), (double) reference(index).real()),
            distance(dst_iec559(index).imag(), (double) reference(index).imag()));
        ++histogram_iec559[ulp_iec559];

        const auto ulp_scaling = std::max(
            distance(dst_scaling(index).real(), (double) reference(index).real()),
            distance(dst_scaling(index).imag(), (double) reference(index).imag()));
        ++histogram_scaling[ulp_scaling];

        const auto ulp_std = std::max(
            distance(dst_std(index).real(), (double) reference(index).real()),
            distance(dst_std(index).imag(), (double) reference(index).imag()));
        ++histogram_std[ulp_std];
    });

    print_histogram("IEC559", histogram_iec559);
    print_histogram("Scaling", histogram_scaling);
    print_histogram("STD", histogram_std);
}

} // namespace reprospect::examples::kokkos::complex
