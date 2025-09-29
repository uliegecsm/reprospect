#ifndef REPROSPECT_ERRORS_CUDA_HPP
#define REPROSPECT_ERRORS_CUDA_HPP

#include <source_location>
#include <sstream>
#include <stdexcept>

#include "cuda_runtime.h"

namespace reprospect::errors
{
//! Check a @c Cuda driver call returns @c cudaError::cudaSuccess.
inline void check_cuda_call(
    const cudaError_enum status,
    const char* const statement,
    const std::source_location& loc = std::source_location::current())
{
    if (status!= cudaError_enum::CUDA_SUCCESS)
    {
        const char* status_name = nullptr;
        const char* status_string = nullptr;
        cuGetErrorName  (status, &status_name);
        cuGetErrorString(status, &status_string);
        std::ostringstream oss;
        oss << statement << " failed: "
            << status << " (" << status_name << ')'
            << status_string
            << " (" << loc.file_name() << ":" << loc.line() << ")";

        throw std::runtime_error(oss.str());
    }
}
} // namespace reprospect::errors

#define REPROSPECT_CHECK_CUDA_CALL(statement) \
    ::reprospect::errors::check_cuda_call((statement), #statement)

#endif // REPROSPECT_ERRORS_CUDA_HPP
