#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "benchmark/benchmark.h"

#include "examples/kokkos/benchmarking.hpp"
#include "examples/kokkos/complex/example_division.hpp"

/**
 * @file
 *
 *  Companion of @ref examples/kokkos/complex/example_division_benchmarking.py.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace reprospect::examples::kokkos::complex {

class Division : public ::benchmark::Fixture {
   public:
    static constexpr unsigned short int range_size = 0;

    static constexpr uint64_t seed = 42;
    static constexpr unsigned short repeat = 5;

    using complex_view_t = Kokkos::View<Kokkos::complex<double>*, Kokkos::CudaSpace>;

    using random_t = Kokkos::Random_XorShift64_Pool<execution_space>;

    template <typename Divisor>
    struct JustDivide {
        typename complex_view_t::const_type src_a, src_b;
        complex_view_t dst_c;

        template <std::integral T>
        KOKKOS_FUNCTION
        void operator()(const T index) const noexcept {
            dst_c(index) = Divisor{}(src_a(index), src_b(index));
        }
    };

   public:
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(TearDown)
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(SetUp)

    void SetUp(const ::benchmark::State& state) override {
        exec.emplace();

        const auto size = state.range(range_size);

        src_a = complex_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec, "src A"), size);
        src_b = complex_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec, "src B"), size);
        dst_c = complex_view_t(Kokkos::view_alloc(*exec, "dst C"), size);

        random_t gen{seed};

        Kokkos::fill_random(*exec, src_a, gen, Kokkos::complex{-50., -50.}, Kokkos::complex{50., 50.});
        Kokkos::fill_random(*exec, src_b, gen, Kokkos::complex{-50., -50.}, Kokkos::complex{50., 50.});

        exec->fence();
    }

    void TearDown(const ::benchmark::State&) override {
        src_a = complex_view_t();
        src_b = complex_view_t();
        dst_c = complex_view_t();
        exec.reset();
    }

    void scaling_branch(const ::benchmark::State& state) const {
        for(unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy(*exec, 0, state.range(range_size)),
            JustDivide<DivisorScalingBranch>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

    void scaling(const ::benchmark::State& state) const {
        for(unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy(*exec, 0, state.range(range_size)),
            JustDivide<DivisorScaling>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

    void iec559(const ::benchmark::State& state) const {
        for(unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy(*exec, 0, state.range(range_size)), JustDivide<DivisorIec559>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

   private:
    std::optional<Kokkos::Cuda> exec = std::nullopt;
    complex_view_t src_a, src_b, dst_c;
};

template <typename Divisor>
class NewtonFractal : public ::benchmark::Fixture {
public:
    using complex_t = Kokkos::complex<double>;
    using view_t = Kokkos::View<unsigned int**, Kokkos::CudaSpace>;
    using roots_t = Kokkos::View<complex_t[3], Kokkos::CudaSpace>;

    static constexpr unsigned short range_width = 0; //! Number of samples along the x axis.
    static constexpr unsigned short range_height = 0; //! Number of samples along the y axis.

    //! Fuction, *e.g.* @c z^3-1.
    KOKKOS_FUNCTION
    static constexpr complex_t function(const complex_t z) {
        return Kokkos::pow(z, 3) - complex_t{1, 0};
    }

    //! Derivative, *e.g.* @c 3*z^2.
    KOKKOS_FUNCTION
    static constexpr complex_t derivative(const complex_t z) {
        return 3 * z * z;
    }

    //! Assume that the domain is [-1, -1] x [-1, 1].
    struct ComputeColors {
        view_t colors;
        view_t iterations;
        roots_t roots;
        unsigned int max_iters;

        double dwidth = 2. / colors.extent(0);
        double dheight = 2. / colors.extent(1);
        double tolerance = 1e-8;

        template <std::integral T>
        KOKKOS_FUNCTION
        void operator()(const T iwidth, const T iheight) const noexcept {
            complex_t z{iwidth * dwidth, iheight * dheight};
            for(unsigned int iter = 0; iter < max_iters; ++iter) {
                z -= Divisor{}(NewtonFractal::function(z), NewtonFractal::derivative(z));
                // for(unsigned int iroot = 0; iroot < roots.size(); ++iroot) {
                //     const complex_t diff = z - roots(iroot);
                //     if(Kokkos::abs(diff.real()) < tolerance && Kokkos::abs(diff.imag()) < tolerance) {
                //         colors(iwidth, iheight) = iroot;
                //         iterations(iwidth, iheight) = iter;
                //         return;
                //     }
                // }
            }
        }
    };

public:
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(TearDown)
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(SetUp)

    void SetUp(const ::benchmark::State& state) {
        exec.emplace();
        colors = view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec), state.range(range_width), state.range(range_height));
        iterations = view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec), state.range(range_width), state.range(range_height));
        roots = roots_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec));
        std::array<complex_t, 3> roots_h{
            complex_t{1., 0.},
            complex_t{-0.5, std::sqrt(3.) / 2.},
            complex_t{-0.5, - std::sqrt(3.) / 2.}
        };
        Kokkos::deep_copy(*exec, roots, roots_t{roots_h.data()});
    }

    void TearDown(const ::benchmark::State&) {
        colors = view_t();
        iterations = view_t();
        roots = roots_t();
        exec.reset();
    }

    void run(const ::benchmark::State& state) const {
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(*exec, {0, 0}, {state.range(range_width), state.range(range_height)}),
            ComputeColors{.colors = colors, .iterations = iterations, .roots = roots, .max_iters = 15000}
        );
        exec->fence();
    }

private:
    std::optional<Kokkos::Cuda> exec = std::nullopt;
    view_t colors, iterations;
    roots_t roots;
};

void parameters(::benchmark::internal::Benchmark* benchmark) {
    benchmark
        ->ArgName(
            "size"
    )
        ->Arg(1024<<1)
        ->Arg(1024<<2)
        ->Arg(1024<<3)
        ->Arg(1024<<4)
        ->Arg(1024<<5)
        ->Arg(1024<<6)
        ->Arg(1024<<7)
        ->Arg(1024<<8)
        ->Arg(1024<<9)
        ->Arg(1024<<10)
        ->Arg(1024<<11)
        ->Arg(1024<<12)
        ->Arg(1024<<13)
        ->Arg(1024<<14)
        ->Arg(1024<<15)
        ->Arg(1024<<16);
}

#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(_name_)                                          \
    BENCHMARK_DEFINE_F(Division, _name_)(benchmark::State & state) {                                  \
        for (auto _: state)                                                                                            \
            this->_name_(state);                                                                                          \
    }                                                                                                                  \
    BENCHMARK_REGISTER_F(Division, _name_)->Apply(parameters);

REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(scaling_branch)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(scaling)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(iec559)

void newton_fractal_parameters(::benchmark::internal::Benchmark* benchmark) {
    benchmark->ArgNames({"width", "height"})
        ->Args({4096, 2300});
}

#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(divisor)                                          \
    BENCHMARK_TEMPLATE_DEFINE_F(NewtonFractal, divisor, Divisor##divisor)(benchmark::State & state) {                                  \
        for (auto _: state)                                                                                            \
            this->run(state);                                                                                          \
    }                                                                                                                  \
    BENCHMARK_REGISTER_F(NewtonFractal, divisor)->Apply(newton_fractal_parameters);

REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(ScalingBranch)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(Scaling)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(Iec559)



} // namespace reprospect::examples::kokkos::complex
