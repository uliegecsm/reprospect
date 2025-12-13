#ifndef REPROSPECT_TESTS_ASSETS_COMMON_CUDA_HPP
#define REPROSPECT_TESTS_ASSETS_COMMON_CUDA_HPP

#include <source_location>
#include <sstream>
#include <stdexcept>

#include "cuda.h"

namespace reprospect::tests
{
//! Check a CUDA driver call returns @c cudaError::cudaSuccess.
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
} // namespace reprospect::tests

#define REPROSPECT_CHECK_CUDA_CALL(statement) \
    ::reprospect::tests::check_cuda_call((statement), #statement)

#endif // REPROSPECT_TESTS_ASSETS_COMMON_CUDA_HPP
