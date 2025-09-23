add_library(cuda_helpers INTERFACE)
target_include_directories(cuda_helpers INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp>)

add_subdirectory(errors)

#---- Install.
install(TARGETS cuda_helpers EXPORT CudaHelpersConfig FILE_SET headers DESTINATION include/cuda-helpers COMPONENT cpp)
install(EXPORT CudaHelpersConfig DESTINATION lib/cmake/CudaHelpers NAMESPACE CudaHelpers:: COMPONENT cpp)
