#include "Kokkos_Core.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/static_batch_size/example_static_batch_size.py.
 */

namespace reprospect::examples::kokkos::static_batch_size {

template <typename ViewType>
struct Increment {
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T) const {
        ++data();
    }
};

//! See https://github.com/kokkos/kokkos/blob/88823f9e368c5308225ad426c3a5935687d90548/core/src/traits/Kokkos_StaticBatchSizeTrait.hpp#L13-L22.
template <unsigned int BatchSize, Kokkos::ExecutionSpace Exec, typename ViewType>
void run(const Exec& exec, ViewType data) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec, Kokkos::Experimental::StaticBatchSize<BatchSize>>(exec, 0, 1),
        Increment{.data = std::move(data)});
}

} // namespace reprospect::examples::kokkos::static_batch_size

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard{argc, argv};
    {
        const Kokkos::Cuda exec{};

        Kokkos::View<unsigned short int, Kokkos::SharedSpace> data(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, exec));

        reprospect::examples::kokkos::static_batch_size::run<1>(exec, data);
        reprospect::examples::kokkos::static_batch_size::run<2>(exec, data);

        exec.fence();
    }
}
