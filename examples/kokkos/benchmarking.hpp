#ifndef REPROSPECT_EXAMPLES_KOKKOS_BENCHMARKING_HPP
#define REPROSPECT_EXAMPLES_KOKKOS_BENCHMARKING_HPP

//! These partial overrides are only needed by 'nvcc' (as of @c Cuda 12.8.0).
#if defined(__NVCC__)
//! Use this macro when only one @c __what__ method is overridden.
#    define FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(__what__)                                                              \
        void __what__(::benchmark::State& state) override {                                                            \
            this->__what__(static_cast<const ::benchmark::State&>(state));                                             \
        }
#else
#    define FIXME_PARTIAL_OVERRIDE_WARNING_NVCC(...)
#endif

#endif // REPROSPECT_EXAMPLES_KOKKOS_BENCHMARKING_HPP
