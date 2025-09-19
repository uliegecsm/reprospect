#ifndef CUDA_HELPERS_HELPERS_HPP
#define CUDA_HELPERS_HELPERS_HPP

#include <source_location>
#include <sstream>
#include <stdexcept>

#include "cuda_runtime.h"

namespace cuda_helpers
{
//! Check a @c Cuda runtime call returns @c cudaError::cudaSuccess.
inline void check_cudart_call(
    cudaError_t status,
    const char* statement,
    const std::source_location& loc = std::source_location::current())
{
    if (status != cudaSuccess)
    {
        std::ostringstream oss;
        oss << statement << " failed: "
            << status << " " << cudaGetErrorString(status)
            << " (" << loc.file_name() << ":" << loc.line() << ")";

        throw std::runtime_error(oss.str());
    }
}
} // namespace cuda_helpers

#define CUDA_HELPERS_CHECK_CUDART_CALL(statement) \
    ::cuda_helpers::check_cudart_call((statement), #statement)

#endif // CUDA_HELPERS_HELPERS_HPP
