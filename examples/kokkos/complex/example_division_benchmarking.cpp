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
            Scaling<true, complex_view_t>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

    void scaling(const ::benchmark::State& state) const {
        for(unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy(*exec, 0, state.range(range_size)),
            Scaling<false, complex_view_t>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

    void iec559(const ::benchmark::State& state) const {
        for(unsigned short irepeat = 0; irepeat < repeat; ++irepeat) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy(*exec, 0, state.range(range_size)), Iec559<complex_view_t>{.src_a = src_a, .src_b = src_b, .dst_c = dst_c});
        }
        exec->fence();
    }

   private:
    std::optional<Kokkos::Cuda> exec = std::nullopt;
    complex_view_t src_a, src_b, dst_c;
};

void parameters(::benchmark::internal::Benchmark* benchmark) {
    benchmark
        ->ArgName(
            "size"
    )
        ->Arg(1024<<1)->Arg(1024<<4)->Arg(1024<<6)->Arg(1024<<8)->Arg(1024<<10);
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

} // namespace reprospect::examples::kokkos::complex
