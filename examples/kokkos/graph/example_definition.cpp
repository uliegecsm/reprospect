#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/graph/example_definition.py.
 */

namespace reprospect::examples::kokkos::graph {

template <typename ViewType>
struct Functor {
    ViewType data;

    KOKKOS_FUNCTION void operator()() const {
        Kokkos::atomic_add(data.data(), typename ViewType::value_type{1});
    }
};

class Definition {
   private:
    using graph_t = Kokkos::Experimental::Graph<Kokkos::Cuda>;
    using view_t = Kokkos::View<int, Kokkos::SharedSpace>;

   public:
    static void run(const Kokkos::Cuda& exec) {
        const view_t data(Kokkos::view_alloc(exec, "data"));

        const auto device_handle = Kokkos::Experimental::get_device_handle(exec);

        std::optional<graph_t> graph = std::nullopt;

        {
            const Kokkos::Profiling::ScopedRegion region("graph - definition");

            graph = Kokkos::Experimental::create_graph(device_handle, [&](const auto& root) {
                const auto node_A = root.then(Functor<view_t>{.data = data});
                Kokkos::Experimental::when_all(
                    node_A.then(Functor<view_t>{.data = data}), node_A.then(Functor<view_t>{.data = data}))
                    .then(Functor<view_t>{.data = data});
            });
        }

        graph->submit(exec);

        exec.fence();

        if (data() != 4)
            throw std::runtime_error("Unexpected value.");
    }
};
} // namespace reprospect::examples::kokkos::graph

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard{argc, argv};
    {
        Kokkos::Profiling::ProfilingSection profiling_section{"definition"};
        profiling_section.start();

        reprospect::examples::kokkos::graph::Definition::run(Kokkos::Cuda{});

        profiling_section.stop();
    }
}
