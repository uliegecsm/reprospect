#include "Kokkos_Core.hpp"

namespace reprospect::examples::kokkos::atomic {

template <typename ViewType>
struct AtomicAddFunctor {
    using scalar_t = typename ViewType::non_const_value_type;

    typename ViewType::non_const_type data;
    scalar_t value;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const {
        Kokkos::atomic_add(&data(index), value);
    }
};

template <typename ViewType>
struct CheckFunctor {
    using scalar_t = typename ViewType::non_const_value_type;

    typename ViewType::const_type data;
    scalar_t value;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index, bool& current) const {
        current = current && (data(index) == value);
    }
};

template <typename ScalarType>
class AtomicAdd {
   public:
    using view_t = Kokkos::View<ScalarType*, Kokkos::CudaSpace>;

   public:
    template <typename T>
    static void run(const Kokkos::Cuda& exec, const T& value) {
        constexpr size_t size = 256;

        const view_t data(Kokkos::view_alloc("data", exec), size);

        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, size), AtomicAddFunctor<view_t>{.data = data, .value = value});

        bool success = false;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy(exec, 0, size),
            CheckFunctor<view_t>{.data = data, .value = value},
            Kokkos::LAnd<bool>(success));

        if (!success)
            throw std::runtime_error("Unexpected failure.");
    }
};

} // namespace reprospect::examples::kokkos::atomic
