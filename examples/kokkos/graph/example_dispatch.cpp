#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

/**
 * @file
 *
 * Companion of @ref examples/kokkos/graph/example_dispatch.py.
 */

namespace reprospect::examples::kokkos::graph
{
template <typename ViewType>
struct Functor
{
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        Kokkos::atomic_add(&data[index], typename ViewType::value_type{1});
    }
};

class Dispatch
{
private:
    using graph_t = Kokkos::Experimental::Graph<Kokkos::Cuda>;
    using view_t  = Kokkos::View<int*, Kokkos::SharedSpace>;

public:
    static void run(const Kokkos::Cuda& exec, const unsigned short nnodes)
    {
        const view_t data(Kokkos::view_alloc(exec, "data"), nnodes);

        graph_t graph{exec};

        const auto root = graph.root_node();

        for(unsigned short inode = 0; inode < nnodes; ++inode)
        {
            root.then_parallel_for(
                Kokkos::RangePolicy(exec, 0, nnodes),
                Functor<view_t>{.data = data}
            );
        }

        graph.instantiate();

        for(unsigned short int isub = 0; isub < 2; ++isub)
        {
            const Kokkos::Profiling::ScopedRegion region("graph - submit - " + std::to_string(isub));
            graph.submit(exec);
        }

        exec.fence("after graph submissions");

        for(unsigned short inode = 0; inode < nnodes; ++inode)
            if(data[inode] != 2 * nnodes)
                throw std::runtime_error("Unexpected value at index " + std::to_string(inode));
    }
};
} // namespace reprospect::examples::kokkos::graph

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        Kokkos::Profiling::ProfilingSection profiling_section {"dispatch"};
        profiling_section.start();

        cudaStream_t stream = nullptr;
        {
            const Kokkos::Profiling::ScopedRegion region("create stream");
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        }

        reprospect::examples::kokkos::graph::Dispatch::run(Kokkos::Cuda{stream}, 5);

        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));

        profiling_section.stop();
    }
}
