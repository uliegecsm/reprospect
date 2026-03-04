#include "Kokkos_Core.hpp"

#include "examples/kokkos/benchmarking.hpp"
#include "examples/kokkos/complex/NewtonFractal.hpp"
#include "examples/kokkos/complex/example_division.hpp"

/**
 * @file
 *
 *  Companion of @ref examples/kokkos/complex/example_division_benchmarking.py.
 */

namespace reprospect::examples::kokkos::complex {

template <typename Divisor>
class NewtonFractal : public ::benchmark::Fixture {
   public:
    static constexpr unsigned short range_width = 0;  //! Number of samples along the x axis.
    static constexpr unsigned short range_height = 1; //! Number of samples along the y axis.

    using value_t = double;

    using function_t = reprospect::examples::kokkos::complex::ZPow4MinOne<value_t, Kokkos::CudaSpace>;
    using compute_t =
        reprospect::examples::kokkos::complex::ComputeColors<value_t, Kokkos::CudaSpace, Divisor, function_t>;

    static constexpr typename compute_t::count_t max_iters = 150;

   public:
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(TearDown)
    FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(SetUp)

    void SetUp(const ::benchmark::State& state) override {
        exec.emplace();
        compute.emplace(*exec, function_t{*exec}, state.range(range_width), state.range(range_height), max_iters);
    }

    void TearDown(const ::benchmark::State&) override {
        compute.reset();
        exec.reset();
    }

    void run(const ::benchmark::State&) const {
        compute->apply(*exec);
        exec->fence();
    }

   private:
    std::optional<Kokkos::Cuda> exec = std::nullopt;
    std::optional<compute_t> compute = std::nullopt;
};

void parameters(::benchmark::internal::Benchmark* benchmark) {
    benchmark->ArgNames({"width", "height"})->Args({128, 128})->Args({512, 512});
}

#define REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_BENCHMARKING(divisor, branching, name)                             \
    BENCHMARK_TEMPLATE_DEFINE_F(NewtonFractal, name, divisor<branching>)(benchmark::State & state) {                   \
        for (auto _: state)                                                                                            \
            this->run(state);                                                                                          \
    }                                                                                                                  \
    BENCHMARK_REGISTER_F(NewtonFractal, name)->Apply(parameters);

REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_BENCHMARKING(DivisorLogbScalbn, false, LogbScalbn)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_BENCHMARKING(DivisorLogbScalbn, true, LogbScalbnBranch)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_BENCHMARKING(DivisorScaling, false, Scaling)
REPROSPECT_EXAMPLES_KOKKOS_COMPLEX_DIVISION_BENCHMARKING(DivisorScaling, true, ScalingBranch)

} // namespace reprospect::examples::kokkos::complex
