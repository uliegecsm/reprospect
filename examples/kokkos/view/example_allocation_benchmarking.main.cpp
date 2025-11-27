#include "benchmark/benchmark.h"

#include "Kokkos_Core.hpp"

/**
 * Steps:
 *  1. Try to disable @c ASLR.
 *  2. Initialize @c Kokkos.
 *  3. Run benchmarks.
 *  4. Finalize @c Kokkos.
 */
int main(int argc, char** argv)
{
    ::benchmark::MaybeReenterWithoutASLR(argc, argv);

    Kokkos::ScopeGuard guard {argc, argv};
    {
        ::benchmark::Initialize(&argc, argv);
        ::benchmark::RunSpecifiedBenchmarks();
        ::benchmark::Shutdown();
    }

    return EXIT_SUCCESS;
}
