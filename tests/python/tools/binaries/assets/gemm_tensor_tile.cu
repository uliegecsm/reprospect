#include <iostream>
#include <filesystem>
#include <format>
#include <fstream>
#include <vector>

#include <mma.h>
#include <cuda_fp16.h>

#include "tests/cpp/cuda/common/cuda_runtime.hpp"

/**
 * @file
 *
 * This is a naïve and under-optimized implementation of @c gemm
 * using @c CUDA tensor cores.
 *
 * References:
 *  * https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html#the-memory-wall
 */

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/**
 * @param M Height of @p mat_A.
 * @param N Width of @p mat_B.
 * @param K Width of @p mat_A and height of @p mat_B.
 *
 * @warning @p M, @p N and @p K must be a multiple of 16 as there is no special handling otherwise.
 *
 * @note It does not use shared memory.
 */
__global__ void gemm_tensor_tile(
    const half * __restrict__ const mat_A,
    const half * __restrict__ const mat_B,
         float * __restrict__ const mat_C,
    const unsigned int M, const unsigned int N, const unsigned int K
) {
    //! Compute linear thread index robustly for 2D block/grid.
    const unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    const unsigned int threads_per_block = blockDim.x * blockDim.y;
    const unsigned int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned int global_thread_id = block_id * threads_per_block + thread_id;

    constexpr unsigned int warp_size = 32;
    const unsigned int warp_id = global_thread_id / warp_size;

    //! Number of WMMA tiles in each dimension (ceiling division).
    const unsigned int num_warp_cols = (N + WMMA_N - 1) / WMMA_N;
    const unsigned int num_warp_rows = (M + WMMA_M - 1) / WMMA_M;
    const unsigned int num_warps_total = num_warp_rows * num_warp_cols;

    if (warp_id >= num_warps_total) return;

    const unsigned int warp_row = warp_id / num_warp_cols;
    const unsigned int warp_col = warp_id % num_warp_cols;

    //! Compute the actual output region this warp handles.
    const unsigned int row_start = warp_row * WMMA_M;
    const unsigned int col_start = warp_col * WMMA_N;

    //! Early exit if this warp is completely out of bounds.
    if (row_start >= M || col_start >= N) return;

    //! Determine actual tile dimensions (may be smaller at edges).
    const unsigned int tile_M = min(WMMA_M, M - row_start);
    const unsigned int tile_N = min(WMMA_N, N - col_start);

    /// Fragment declarations.
    /// For row-major A (MxK) and row-major B (KxN), we need:
    /// - A fragment, row-major
    /// - B fragment, row-major
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    //! Initialize accumulator to zero.
    wmma::fill_fragment(c_frag, 0.f);

    /// Loop over K dimension in WMMA_K-sized tiles.
    /// @warning This assumes K is a multiple of WMMA_K. If not, padding is required.
    for (unsigned int tile_K = 0; tile_K < K; tile_K += WMMA_K)
    {
        const half* const tile_A = mat_A + (size_t)row_start * K + tile_K;
        const half* const tile_B = mat_B + (size_t)tile_K    * N + col_start;

        /// Leading dimensions (stride between consecutive rows).
        /// A is MxK in row-major.
        /// B is KxN in row-major.
        const int lda = K;
        const int ldb = N;

        //! Load and multiply-accumulate.
        wmma::load_matrix_sync(a_frag, tile_A, lda);
        wmma::load_matrix_sync(b_frag, tile_B, ldb);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    //! Store result to global memory.
    float * const tileC = mat_C + (size_t)row_start * N + col_start;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

//! Read vector from binary file.
template <typename T>
auto read_from_binary_file(const std::filesystem::path& file, const std::size_t count) -> std::vector<T>
{
    std::ifstream fin(file, std::ios::binary);
    if (!fin)
        throw std::runtime_error(std::format("Failed to open file {}.", file.c_str()));

    std::vector<T> data(count);
    if (!fin.read(reinterpret_cast<char*>(data.data()), count * sizeof(T)))
        throw std::runtime_error(std::format("Failed to read {} elements from {}.", count, file.c_str()));

    return data;
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0] << " M N K A.bin B.bin C.bin" << std::endl;
        return 1;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);

    const char* const file_A = argv[4];
    const char* const file_B = argv[5];
    const char* const file_C = argv[6];

    //! Read matrices A and B.
    const auto mat_A_h = read_from_binary_file<half>(file_A, M * K);
    const auto mat_B_h = read_from_binary_file<half>(file_B, K * N);

    //! Create a stream.
    cudaStream_t stream = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamCreate(&stream));

    //! Allocate device memory.
    half* mat_A_d = nullptr;
    half* mat_B_d = nullptr;
    float* mat_C_d = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&mat_A_d, sizeof(half)  * M * K, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&mat_B_d, sizeof(half)  * K * N, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaMallocAsync(&mat_C_d, sizeof(float) * M * N, stream));

    REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(mat_A_d, mat_A_h.data(), sizeof(half)  * M * K, cudaMemcpyHostToDevice, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(mat_B_d, mat_B_h.data(), sizeof(half)  * K * N, cudaMemcpyHostToDevice, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaMemsetAsync(mat_C_d, 0,              sizeof(float) * M * N,                         stream));

    //! Launch kernel.
    const int warps = (M / WMMA_M) * (N / WMMA_N);
    constexpr int threads_per_block = 128;
    const int blocks = (warps * 32 + threads_per_block - 1) / threads_per_block;

    gemm_tensor_tile<<<blocks, threads_per_block, 0, stream>>>(mat_A_d, mat_B_d, mat_C_d, M, N, K);
    REPROSPECT_CHECK_CUDART_CALL(cudaGetLastError());

    //! Copy matrix C back to host and write to file.
    std::vector<float> mat_C_h(M * N);
    REPROSPECT_CHECK_CUDART_CALL(cudaMemcpyAsync(mat_C_h.data(), mat_C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost, stream));

    {
        std::ofstream fout(file_C, std::ios::binary);
        fout.write(reinterpret_cast<char*>(mat_C_h.data()), M * N * sizeof(float));
    }

    //! Free resources.
    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(mat_A_d, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(mat_B_d, stream));
    REPROSPECT_CHECK_CUDART_CALL(cudaFreeAsync(mat_C_d, stream));

    REPROSPECT_CHECK_CUDART_CALL(cudaStreamDestroy(stream));

    return EXIT_SUCCESS;
}
