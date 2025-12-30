#include "Kokkos_Core.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

namespace reprospect::examples::kokkos::view
{
#define OPERATION(dst, src_a, src_b, index)    \
    dst[index]  = src_a[index] + src_b[index]; \
    dst[index] += src_a[index] + src_b[index];

template <typename T>
__global__ void global_kernel(
    T* __restrict__ const dst,
    const T* __restrict__ const src_a, const T* __restrict__ const src_b,
    const unsigned int size
) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        OPERATION(dst, src_a, src_b, index)
    }
}

template <typename T>
struct FunctorRestrictMember
{
    const T* __restrict__ src_a;
    const T* __restrict__ src_b;
          T* __restrict__ dst;

    template <std::integral U>
    KOKKOS_FUNCTION void operator()(const U index) const {
        OPERATION(dst, src_a, src_b, index)
    }
};

template <typename T>
struct FunctorRestrictRecastLambda
{
    const T* src_a;
    const T* src_b;
          T* dst;

    template <std::integral U>
    KOKKOS_FUNCTION void operator()(const U index) const
    {
        [index](T* __restrict__ const d, const T* __restrict__ const a, const T* __restrict__ const b) {
            OPERATION(d, a, b, index)
        }(dst, src_a, src_b);
    }
};

template <typename T>
struct FunctorRestrictRecastLocal
{
    const T* src_a;
    const T* src_b;
          T* dst;

    template <std::integral U>
    KOKKOS_FUNCTION void operator()(const U index) const
    {
              T* __restrict__ const dst_r = this->dst;
        const T* __restrict__ const src_a_r = this->src_a;
        const T* __restrict__ const src_b_r = this->src_b;
        OPERATION(dst_r, src_a_r, src_b_r, index)
    }
};

template <typename T>
struct FunctorRestrictViewMemoryTrait
{
    using view_t = Kokkos::View<T*, Kokkos::CudaSpace, Kokkos::MemoryTraits<Kokkos::Restrict>>;

    typename view_t::const_type src_a, src_b;
    view_t dst;

    template <std::integral U>
    KOKKOS_FUNCTION void operator()(const U index) const {
        OPERATION(dst, src_a, src_b, index)
    }
};

//! Inspired by https://github.com/kokkos/kokkos/blob/37f70304dc3676691af88d3ac3ba50cddbfa337f/core/src/Cuda/Kokkos_Cuda_View.hpp#L17-L26.
template <typename T> requires (sizeof(T) == 4)
struct LDGAccessor
{
    const T* m_ptr;

    template <typename U>
    KOKKOS_FUNCTION T operator[](const U index) const {
        return __ldg(&m_ptr[index]);
    }
};

template <typename T>
struct FunctorLDGAccessor
{
    LDGAccessor<const T> src_a, src_b;
    T* dst;

    template <std::integral U>
    KOKKOS_FUNCTION void operator()(const U index) const {
        OPERATION(dst, src_a, src_b, index)
    }
};

template <typename T>
struct RestrictAccessor
{
    using bare_reference = T&;
    using reference      = bare_reference KOKKOS_RESTRICT;

    T* m_ptr;

    template <typename U>
    KOKKOS_FUNCTION reference operator[](const U index) const {
        return m_ptr[index];
    }
};

template <typename T>
struct FunctorRestrictAccessor
{
    RestrictAccessor<const T> src_a, src_b;
    RestrictAccessor<      T> dst;

    template <std::integral U>
    KOKKOS_FUNCTION void operator()(const U index) const {
        OPERATION(dst, src_a, src_b, index)
    }
};

template <typename ViewType>
struct FunctorCheck
{
    typename ViewType::const_type data;
    typename ViewType::value_type value;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index, bool& current) const {
        current = current && data[index] == value;
    }
};

class Restrict
{
public:
    using scalar_t = int;
    using view_t = Kokkos::View<scalar_t*, Kokkos::CudaSpace>;

    static constexpr size_t size = 128;

public:
    Restrict(const Kokkos::Cuda& exec)
      : src_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A", exec), size),
        src_b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "B", exec), size),
        dst(Kokkos::view_alloc("D", exec), size) {}

    void reset(const Kokkos::Cuda& exec) const
    {
        Kokkos::deep_copy(exec, src_a, 1);
        Kokkos::deep_copy(exec, src_b, 2);
    }

    void check(const Kokkos::Cuda& exec) const
    {
        bool success = false;
        Kokkos::parallel_reduce(Kokkos::RangePolicy(exec, 0, size),
            FunctorCheck<view_t>{.data = dst, .value = 2 * (1 + 2)},
            Kokkos::LAnd<bool>(success)
        );
        if(!success)
            throw std::runtime_error("something went wrong");
    }

    void global_kernel(const Kokkos::Cuda& exec) const
    {
        const Kokkos::Profiling::ScopedRegion region{"global_kernel"};
        ::reprospect::examples::kokkos::view::global_kernel<<<1, size, 0, exec.cuda_stream()>>>(
            dst.data(), src_a.data(), src_b.data(), size
        );
    }

    void restrict_recast_lambda(const Kokkos::Cuda& exec) const
    {
        Kokkos::parallel_for("restrict_recast_lambda", Kokkos::RangePolicy(exec, 0, size),
            FunctorRestrictRecastLambda<scalar_t>{
                .src_a = src_a.data(), .src_b = src_b.data(), .dst = dst.data()
            }
        );
    }

    void restrict_recast_local(const Kokkos::Cuda& exec) const
    {
        Kokkos::parallel_for("restrict_recast_local", Kokkos::RangePolicy(exec, 0, size),
            FunctorRestrictRecastLocal<scalar_t>{
                .src_a = src_a.data(), .src_b = src_b.data(), .dst = dst.data()
            }
        );
    }

    void restrict_accessor(const Kokkos::Cuda& exec) const
    {
        Kokkos::parallel_for("restrict_accessor", Kokkos::RangePolicy(exec, 0, size),
            FunctorRestrictAccessor<scalar_t>{
                .src_a = {.m_ptr = src_a.data()}, .src_b = {.m_ptr = src_b.data()}, .dst = {.m_ptr = dst.data()}
            }
        );
    }

    void restrict_member(const Kokkos::Cuda& exec) const
    {
        Kokkos::parallel_for("restrict_member", Kokkos::RangePolicy(exec, 0, size),
            FunctorRestrictMember<scalar_t>{
                .src_a = src_a.data(), .src_b = src_b.data(), .dst = dst.data()
            }
        );
    }

    void restrict_view_memory_trait(const Kokkos::Cuda& exec) const
    {
        Kokkos::parallel_for("restrict_view_memory_trait", Kokkos::RangePolicy(exec, 0, size),
            FunctorRestrictViewMemoryTrait<scalar_t>{
                .src_a = src_a, .src_b = src_b, .dst = dst
            }
        );
    }

    void ldg_accessor(const Kokkos::Cuda& exec) const
    {
        Kokkos::parallel_for("ldg_accessor", Kokkos::RangePolicy(exec, 0, size),
            FunctorLDGAccessor<scalar_t>{
                .src_a = {.m_ptr = src_a.data()}, .src_b = {.m_ptr = src_b.data()}, .dst = dst.data()
            }
        );
    }

protected:
    view_t src_a;
    view_t src_b;
    view_t dst;
};
} // namespace reprospect::examples::kokkos::view

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard {argc, argv};
    {
        const Kokkos::Cuda exec {};

        const auto tester = ::reprospect::examples::kokkos::view::Restrict{exec};

        Kokkos::Profiling::ProfilingSection section{"restrict"};
        section.start();

#define RUN_ONE_CASE(exec, case) \
    tester.reset(exec);          \
    tester.case(exec);           \
    tester.check(exec);

        RUN_ONE_CASE(exec, global_kernel);

        RUN_ONE_CASE(exec, restrict_recast_lambda);
        RUN_ONE_CASE(exec, restrict_recast_local);

        RUN_ONE_CASE(exec, restrict_accessor);
        RUN_ONE_CASE(exec, restrict_member);
        RUN_ONE_CASE(exec, restrict_view_memory_trait);

        RUN_ONE_CASE(exec, ldg_accessor);

        section.stop();
    }
}
