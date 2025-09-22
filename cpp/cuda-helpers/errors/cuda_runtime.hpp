#ifndef CUDA_HELPERS_ERRORS_CUDA_RUNTIME_HPP
#define CUDA_HELPERS_ERRORS_CUDA_RUNTIME_HPP

#include <source_location>
#include <sstream>
#include <stdexcept>

#include "cuda_runtime.h"

namespace cuda_helpers::errors
{
//! Check a @c Cuda runtime call returns @c cudaError::cudaSuccess.
inline void check_cudart_call(
    const cudaError_t status,
    const char* const statement,
    const std::source_location& loc = std::source_location::current())
{
    if (status != cudaSuccess)
    {
        std::ostringstream oss;
        oss << statement << " failed: "
            << status << " (" << cudaGetErrorName(status) << ')'
            << cudaGetErrorString(status)
            << " (" << loc.file_name() << ":" << loc.line() << ")";

        throw std::runtime_error(oss.str());
    }
}
} // namespace cuda_helpers::errors

#define CUDA_HELPERS_CHECK_CUDART_CALL(statement) \
    ::cuda_helpers::errors::check_cudart_call((statement), #statement)

#endif // CUDA_HELPERS_ERRORS_CUDA_RUNTIME_HPP
