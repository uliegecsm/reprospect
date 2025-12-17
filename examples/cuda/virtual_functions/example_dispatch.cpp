#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "cuda/runtime_helper.hpp"
#include "nvtx3/nvtx3.hpp"

/**
 * @file
 *
 * Companion of @ref examples/cuda/virtual_functions/example_dispatch.py.
 */

namespace reprospect::examples::cuda::virtual_functions
{

/**
 * @name Utilities for dispatching virtual functions in device kernels.
 *
 * The main challenge when dealing with virtual functions in device kernels is ensuring that the correct
 * vtable is used on device. This requires that the derived class instance is constructed on device.
 *
 * The following utilities facilitate this by providing a way to copy construct a device object
 * from a host object. This is achieved by allocating raw memory on device and using
 * a placement new within a device kernel to construct the device object in that memory. A custom
 * deleter is provided to a @c std::shared_ptr to ensure that the device object is properly
 * destructed on device and the memory is freed when the @c std::shared_ptr goes out of scope.
 *
 * References:
 *
 * * @cite brunini-2019-runtime-polymorphism-kokkos
 * * @cite howard-2017-towards-performance-portability-cfd
 * * https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#polymorphic-classes
 * * https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#implicitly-declared-and-non-virtual-explicitly-defaulted-functions
 */
///@{
template <typename T>
__global__ void __launch_bounds__(1, 1) delete_kernel(T* const ptr) {
    if (blockIdx.x == 0) ptr->~T();
}

struct DeviceDeleter
{
    template <typename T>
    void operator()(T* const ptr) const
    {
        delete_kernel<<<1, 1>>>(ptr);
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(nullptr));
        REPROSPECT_CHECK_CUDART_CALL(cudaFree(ptr));
    }
};

template <typename Derived>
__global__ void __launch_bounds__(1, 1) copy_construct_kernel(Derived* const ptr, const Derived derived) {
    if (blockIdx.x == 0) new (ptr) Derived(derived);
}

template <typename Derived>
std::shared_ptr<Derived> copy_to_device(cudaStream_t stream, const Derived& derived)
{
    Derived* ptr = nullptr;
    REPROSPECT_CHECK_CUDART_CALL(cudaMalloc(&ptr, sizeof(Derived)));

    copy_construct_kernel<<<1, 1, 0, stream>>>(ptr, derived);

    return {ptr, DeviceDeleter{}};
}
///@}

struct Base
{
    __device__ virtual void foo(const unsigned int /* */) const = 0;
    __device__ virtual void bar(const unsigned int /* */) const = 0;

    __host__ __device__ virtual ~Base() {}
};

template <typename T>
struct DerivedA : public Base
{
    T* x;

    DerivedA(T* x_) : x(x_) {}

    __device__ void foo(const unsigned int idx) const override { x[idx] += 0xaf; }
    __device__ void bar(const unsigned int idx) const override { x[idx] += 0xab; }
};

template <typename T>
struct DerivedB : public Base
{
    T* x;

    DerivedB(T* x_) : x(x_) {}

    __device__ void foo(const unsigned int idx) const override { x[idx] += 0xbf; }
    __device__ void bar(const unsigned int idx) const override { x[idx] += 0xbb; }
};

template <typename Derived>
__global__ void static_foo_kernel(const Base* const base, const unsigned int size)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        static_cast<const Derived*>(base)->Derived::foo(index);
    }
}

__global__ void dynamic_foo_kernel(const Base* const base, const unsigned int size)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        base->foo(index);
    }
}

__global__ void dynamic_bar_kernel(const Base* const base, const unsigned int size)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        base->bar(index);
    }
}

__global__ void dynamic_foo_bar_kernel(const Base* const base, const unsigned int size)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        base->foo(index);
        base->bar(index);
    }
}

class Dispatch
{
public:
    using value_t = float;

    static constexpr unsigned int size = 128;

    using base_t      = Base;
    using derived_a_t = DerivedA<value_t>;
    using derived_b_t = DerivedB<value_t>;

public:
    Dispatch(cudaStream_t stream)
      : generator(std::random_device{}())
    {
        //! Create and initialize buffer.
        REPROSPECT_CHECK_CUDART_CALL(cudaMalloc(&x, size * sizeof(value_t)));

        REPROSPECT_CHECK_CUDART_CALL(cudaMemsetAsync(x, 0, size * sizeof(value_t), stream));

        //! Create host and device objects.
        const derived_a_t derived_a_h(x);
        derived_a_sdptr = copy_to_device(stream, derived_a_h);

        const derived_b_t derived_b_h(x);
        derived_b_sdptr = copy_to_device(stream, derived_b_h);
    }

    ~Dispatch() {
        REPROSPECT_CHECK_CUDART_CALL(cudaFree(x));
    }

    void run_static_foo(cudaStream_t stream) const
    {
        const bool use_a = draw();
        {
            nvtx3::scoped_range range("static_foo");
            if (use_a) {
                static_foo_kernel<derived_a_t><<<1, size, 0, stream>>>(derived_a_sdptr.get(), size);
            } else {
                static_foo_kernel<derived_b_t><<<1, size, 0, stream>>>(derived_b_sdptr.get(), size);
            }
        }
        check(stream, use_a ? 0xaf : 0xbf);
    }

    void run_dynamic_foo(cudaStream_t stream) const
    {
        const bool use_a = draw();
        {
            nvtx3::scoped_range range("dynamic_foo");
            const base_t* const base_rdptr = use_a ? derived_a_sdptr.get() : derived_b_sdptr.get();
            dynamic_foo_kernel<<<1, size, 0, stream>>>(base_rdptr, size);
        }
        check(stream, use_a ? 0xaf : 0xbf);
    }

    void run_dynamic_bar(cudaStream_t stream) const
    {
        const bool use_a = draw();
        {
            nvtx3::scoped_range range("dynamic_bar");
            const base_t* const base_rdptr = use_a ? derived_a_sdptr.get() : derived_b_sdptr.get();
            dynamic_bar_kernel<<<1, size, 0, stream>>>(base_rdptr, size);
        }
        check(stream, use_a ? 0xab : 0xbb);
    }

    void run_dynamic_foo_bar(cudaStream_t stream) const
    {
        const bool use_a = draw();
        {
            nvtx3::scoped_range range("dynamic_foo_bar");
            const base_t* const base_rdptr = use_a ? derived_a_sdptr.get() : derived_b_sdptr.get();
            dynamic_foo_bar_kernel<<<1, size, 0, stream>>>(base_rdptr, size);
        }
        check(stream, use_a ? 0xaf + 0xab : 0xbf + 0xbb);
    }

    bool draw() const {
        return std::bernoulli_distribution{0.5}(generator);
    }

    void check(cudaStream_t stream, const value_t expt_val) const {
        std::vector<value_t> x_h(size);
        REPROSPECT_CHECK_CUDART_CALL(
            cudaMemcpyAsync(x_h.data(), x, size * sizeof(value_t), cudaMemcpyDeviceToHost, stream)
        );
        REPROSPECT_CHECK_CUDART_CALL(cudaStreamSynchronize(stream));

        bool all_as_expected = true;
        for (unsigned int index = 0; index < size; ++index)
        {
            if (x_h[index] != expt_val) {
                all_as_expected = false;
                break;
            }
        }

        if (!all_as_expected) {
            throw std::runtime_error("buffer elements not as expected");
        }
    }

protected:
    value_t* x;
    std::shared_ptr<const base_t> derived_a_sdptr, derived_b_sdptr;
    mutable std::mt19937 generator;
};

} // namespace reprospect::examples::cuda::virtual_functions

int main()
{
    using namespace reprospect::examples::cuda::virtual_functions;

    cudaStream_t stream;
    REPROSPECT_CHECK_CUDART_CALL(cudaStreamCreate(&stream));

    nvtxRangePush("Dispatch");

    Dispatch{stream}.run_static_foo     (stream);
    Dispatch{stream}.run_dynamic_foo    (stream);
    Dispatch{stream}.run_dynamic_bar    (stream);
    Dispatch{stream}.run_dynamic_foo_bar(stream);

    nvtxRangePop();

    REPROSPECT_CHECK_CUDART_CALL(cudaStreamDestroy(stream));

    return EXIT_SUCCESS;
}
