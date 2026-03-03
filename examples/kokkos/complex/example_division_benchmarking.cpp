#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "benchmark/benchmark.h"

#include "examples/kokkos/benchmarking.hpp"
#include "examples/kokkos/complex/NewtonFractal.hpp"
#include "examples/kokkos/complex/example_division.hpp"

#include <fstream>
#include <iostream>
#include <type_traits>

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
        KOKKOS_FUNCTION void operator()(const T index) const noexcept {
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
        for (unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy(*exec, 0, state.range(range_size)),
                JustDivide<DivisorScalingBranch>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

    void scaling(const ::benchmark::State& state) const {
        for (unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy(*exec, 0, state.range(range_size)),
                JustDivide<DivisorScaling>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

    void iec559(const ::benchmark::State& state) const {
        for (unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy(*exec, 0, state.range(range_size)),
                JustDivide<DivisorIec559>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

   private:
    std::optional<Kokkos::Cuda> exec = std::nullopt;
    complex_view_t src_a, src_b, dst_c;
};

template <class ViewType>
void dump_view_binary(const ViewType& view, const std::string& filename) {
    using value_type = typename ViewType::value_type;

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    std::cout << "sizeof(value_type): " << sizeof(value_type) << std::endl;

    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    const size_t dim0 = view_h.extent(0);
    const size_t dim1 = view_h.extent(1);

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file");
    }

    out.write(reinterpret_cast<const char*>(&dim0), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&dim1), sizeof(size_t));

    out.write(reinterpret_cast<const char*>(view_h.data()), sizeof(value_type) * dim0 * dim1);

    out.close();
}

template <typename Divisor, typename RealType = double>
class NewtonFractal : public ::benchmark::Fixture {
   public:
    static constexpr unsigned short range_width = 0;  //! Number of samples along the x axis.
    static constexpr unsigned short range_height = 1; //! Number of samples along the y axis.

    using function_t = ZPow4MinOne<Kokkos::CudaSpace, RealType>;
    using compute_t = ComputeColors<Kokkos::CudaSpace, RealType, Divisor, function_t>;

   public:
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(TearDown)
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(SetUp)

    void SetUp(const ::benchmark::State& state) {
        exec.emplace();
        const auto width = state.range(range_width);
        const auto height = state.range(range_height);
        std::cout << "> width x height: " << width << "x" << height << std::endl;
        colors =
            typename compute_t::count_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec), width, height);
        iterations =
            typename compute_t::count_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, *exec), width, height);
        function = function_t{*exec};
    }

    void TearDown(const ::benchmark::State&) {
        dump_view_binary(colors, "colors.bin");
        dump_view_binary(iterations, "iterations.bin");
        colors = typename compute_t::count_view_t();
        iterations = typename compute_t::count_view_t();
        function = function_t{};
        exec.reset();
    }

    void run(const ::benchmark::State& state) const {
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                *exec, {0, 0}, {state.range(range_width), state.range(range_height)}),
            compute_t{.function = function, .colors = colors, .iterations = iterations, .max_iters = 150});
        exec->fence();
    }

   private:
    std::optional<Kokkos::Cuda> exec = std::nullopt;
    typename compute_t::count_view_t colors, iterations;
    function_t function;
};

void parameters(::benchmark::internal::Benchmark* benchmark) {
    benchmark
        ->ArgName("size")
        // ->Arg(1024<<1)
        ->Arg(1024 << 2)
        // ->Arg(1024<<3)
        ->Arg(1024 << 4)
        // ->Arg(1024<<5)
        ->Arg(1024 << 6)
        // ->Arg(1024<<7)
        ->Arg(1024 << 8)
        // ->Arg(1024<<9)
        ->Arg(1024 << 10)
        // ->Arg(1024<<11)
        ->Arg(1024 << 12)
        // ->Arg(1024<<13)
        ->Arg(1024 << 14)
        // ->Arg(1024<<15)
        ->Arg(1024 << 16);
}

#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(_name_)                                                            \
    BENCHMARK_DEFINE_F(Division, _name_)(benchmark::State & state) {                                                   \
        for (auto _: state)                                                                                            \
            this->_name_(state);                                                                                       \
    }                                                                                                                  \
    BENCHMARK_REGISTER_F(Division, _name_)->Apply(parameters);

REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(scaling_branch)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(scaling)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION(iec559)

void newton_fractal_parameters(::benchmark::internal::Benchmark* benchmark) {
    benchmark->ArgNames({"width", "height"})->Args({4096, 2300});
}

#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(divisor)                                            \
    BENCHMARK_TEMPLATE_DEFINE_F(NewtonFractal, divisor, Divisor##divisor)(benchmark::State & state) {                  \
        for (auto _: state)                                                                                            \
            this->run(state);                                                                                          \
    }                                                                                                                  \
    BENCHMARK_REGISTER_F(NewtonFractal, divisor)->Apply(newton_fractal_parameters);

REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(ScalingBranch)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(Scaling)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_NEWTON_FRACTAL(Iec559)


} // namespace reprospect::examples::kokkos::complex
