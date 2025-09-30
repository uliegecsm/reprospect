#include <array>
#include <stdexcept>
#include <vector>

#include "common/cuda_runtime.hpp"
#include "cub/detail/nvtx3.hpp"

/**
 * @file
 *
 * A simple diamond graph.
 */

using index_t = decltype(dim3::x);

template <index_t Dst, index_t... Src>
__global__ __launch_bounds__(1, 1) void add_and_increment_kernel(index_t* const counters)
{
    ((counters[Dst] += counters[Src]), ...);
    ++counters[Dst];
}

template <index_t Dst, size_t NumPredecessors, index_t... Src>
cudaGraphNode_t add_node(const cudaGraph_t graph, index_t* const counters, const std::array<const cudaGraphNode_t, NumPredecessors> predecessors)
{
    cudaGraphNode_t node = nullptr;

    void* args[] = {(void*)&counters};

    const cudaKernelNodeParams params = {
        .func = (void*)add_and_increment_kernel<Dst, Src...>,
        .gridDim = dim3(1),
        .blockDim = dim3(1),
        .sharedMemBytes = 0,
        .kernelParams = args,
        .extra = nullptr
    };

    REPROSPECT_CHECK_CUDART_CALL(cudaGraphAddKernelNode(&node, graph, predecessors.empty() ? nullptr : predecessors.data(), predecessors.size(), &params));
    return node;
}

struct MyAppDomain{ static constexpr char const* name {"application-domain"}; };

int main()
{
    //! Mark the start of the application.
    ::nvtx3::mark_in<MyAppDomain>("Starting my application.");

    //! This one is superfluous but serves the tests.
    const auto& outer = ::nvtx3::start_range_in<MyAppDomain>("outer-useless-range");

    //! Create stream.
    cudaStream_t stream = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamCreate(&stream));

    //! Allocate data.
    index_t* counters = nullptr;

    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&counters, 4 * sizeof(index_t), stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaMemsetAsync(counters, 0, 4 * sizeof(index_t), stream));

    //! Create graph.
    cudaGraph_t graph = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaGraphCreate(&graph, 0));

    //! Node A increments index 0.
    const auto node_A = add_node<0, 0>(graph, counters, {});

    //! Node B depends on node A. It increments index 1 and adds the value at index 0 to index 1.
    const auto node_B = add_node<1, 1, 0>(graph, counters, {node_A});

    //! Node C depends on node A. It increments index 2 and adds the value at index 0 to index 2.
    const auto node_C = add_node<2, 1, 0>(graph, counters, {node_A});

    /// Node D depends on nodes B and C. It increments index 3 and adds
    /// the values at indices 1 and 2 to index 3.
    const auto node_D = add_node<3, 2, 1, 2>(graph, counters, {node_B, node_C});

    //! Instantiate and launch the graph.
    cudaGraphExec_t graph_exec = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    REPROSPECT_CHECK_CUDART_CALL(cudaGraphLaunch(graph_exec, stream));

    std::vector<index_t> counters_h(4);
    REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(counters_h.data(), counters, 4 * sizeof(index_t), cudaMemcpyDeviceToHost, stream));

    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(counters, stream));

    REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));

    REPROSPECT_CHECK_CUDART_CALL(cudaGraphExecDestroy(graph_exec));
    REPROSPECT_CHECK_CUDART_CALL(cudaGraphDestroy(graph));
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamDestroy(stream));

    if(counters_h.at(0) != 1) throw std::runtime_error("wrong value for node A");
    if(counters_h.at(1) != 2) throw std::runtime_error("wrong value for node B");
    if(counters_h.at(2) != 2) throw std::runtime_error("wrong value for node C");
    if(counters_h.at(3) != 5) throw std::runtime_error("wrong value for node D");

    ::nvtx3::end_range_in<MyAppDomain>(outer);

    return 0;
}
