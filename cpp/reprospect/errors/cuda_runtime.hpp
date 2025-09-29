#ifndef REPROSPECT_ERRORS_CUDA_RUNTIME_HPP
#define REPROSPECT_ERRORS_CUDA_RUNTIME_HPP

#include <source_location>
#include <sstream>
#include <stdexcept>

#include "cuda_runtime.h"

namespace reprospect::errors
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
} // namespace reprospect::errors

#define REPROSPECT_CHECK_CUDART_CALL(statement) \
    ::reprospect::errors::check_cudart_call((statement), #statement)

#endif // REPROSPECT_ERRORS_CUDA_RUNTIME_HPP
