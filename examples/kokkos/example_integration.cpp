#include "Kokkos_Core.hpp"

struct SayHello
{
    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        Kokkos::printf("Hello from index %d.\n", index);
    }
};

int main()
{
    Kokkos::ScopeGuard guard {};
    {
        const Kokkos::DefaultExecutionSpace exec {};
        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, 42),
            SayHello{}
        );
        exec.fence();
    }
}
