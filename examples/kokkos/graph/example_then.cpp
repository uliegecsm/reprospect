#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/graph/example_then.py.
 */

namespace reprospect::examples::kokkos::graph
{
template <typename ViewType>
struct Functor
{
    ViewType data;

    KOKKOS_FUNCTION
    void operator()() const {
        ++data();
    }
};

struct Then
{
public:
    using graph_t = Kokkos::Experimental::Graph<Kokkos::Cuda>;
    using view_t  = Kokkos::View<int, Kokkos::SharedSpace>;

public:
    static void run(const Kokkos::Cuda& exec)
    {
        const view_t data(Kokkos::view_alloc(exec, "data"));

        const graph_t graph {exec};

        graph.root_node().then("then", exec, Functor<view_t>{.data = data});

        graph.submit(exec);

        exec.fence();

        if(data() != 1)
            throw std::runtime_error("Unexpected value.");
    }
};
} // namespace reprospect::examples::kokkos::graph

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        Kokkos::Profiling::ProfilingSection profiling_section {"then"};
        profiling_section.start();

        reprospect::examples::kokkos::graph::Then::run(Kokkos::Cuda{});

        profiling_section.stop();
    }
}
