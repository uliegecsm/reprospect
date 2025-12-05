#include "Kokkos_Core.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

#if !defined(KOKKOS_ENABLE_COMPLEX_ALIGN)
    #error "KOKKOS_ENABLE_COMPLEX_ALIGN is not defined."
#endif

/**
 * @file
 *
 * Companion of @ref examples/kokkos/complex/example_alignment.py.
 */

namespace reprospect::examples::kokkos::complex
{
template <typename T>
struct Complex
{
    using value_type = T;

    value_type real_, imag_;

    KOKKOS_FUNCTION constexpr Complex& operator+=(
        const Complex& other
    ) noexcept {
        real_ += other.real_;
        imag_ += other.imag_;
        return *this;
    }

    KOKKOS_FUNCTION friend constexpr Complex operator*(
        const Complex& src_a, const Complex& src_b
    ) noexcept { return {
            src_a.real_ * src_b.real_ - src_a.imag_ * src_b.imag_,
            src_a.real_ * src_b.imag_ + src_a.imag_ * src_b.real_
        };
    }
};

template <typename ViewType>
struct MulAddFunctor
{
    typename ViewType::const_type src_a, src_b;
    ViewType dst_c;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T idx) const {
        dst_c(idx) += src_a(idx) * src_b(idx);
    }
};

template <typename ComplexType>
class AlignmentProfiling
{
public:
    static constexpr size_t size = 1024;

    using complex_t      = ComplexType;
    using complex_view_t = Kokkos::View<complex_t*, Kokkos::CudaSpace>;

public:
    static void run(const Kokkos::Cuda& exec)
    {
        const complex_view_t values_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "values A", exec), size);
        const complex_view_t values_b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "values B", exec), size);
        const complex_view_t values_c(Kokkos::view_alloc(                             "values C", exec), size);

        Kokkos::deep_copy(exec, values_a, {4, 2});
        Kokkos::deep_copy(exec, values_b, {2, 1});

        const Kokkos::Profiling::ScopedRegion region(
            std::alignment_of_v<complex_t> == sizeof(typename complex_t::value_type)
                ? "default"
                : "specified"
        );

        Kokkos::parallel_for(
            "multiply and add view elements",
            Kokkos::RangePolicy(exec, 0, size),
            MulAddFunctor{.src_a = values_a, .src_b = values_b, .dst_c = values_c}
        );

        exec.fence();
    }
};

} // namespace reprospect::examples::kokkos::complex

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        using namespace reprospect::examples::kokkos::complex;

        Kokkos::Profiling::ProfilingSection profiling_section {"AlignmentProfiling"};
        profiling_section.start();

        using complex_double_specified_alignment_t = Kokkos::complex<double>;
        static_assert(std::alignment_of_v<complex_double_specified_alignment_t> == sizeof(double) * 2);
        static_assert(sizeof(complex_double_specified_alignment_t) == sizeof(double) * 2);
        AlignmentProfiling<complex_double_specified_alignment_t>::run(Kokkos::Cuda{});

        using complex_double_t = Complex<double>;
        static_assert(std::alignment_of_v<complex_double_t> == sizeof(double));
        static_assert(sizeof(complex_double_t) == sizeof(double) * 2);
        AlignmentProfiling<complex_double_t>::run(Kokkos::Cuda{});

        profiling_section.stop();
    }
}
