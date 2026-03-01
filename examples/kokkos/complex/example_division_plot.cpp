#include <filesystem>
#include <fstream>

#include "examples/kokkos/complex/NewtonFractal.hpp"
#include "examples/kokkos/complex/example_division.hpp"

template <typename ViewType>
void dump(const ViewType& view, const std::filesystem::path& filename) {
    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file " + filename.string());
    }

    Kokkos::printf("Writing view %s to %s.\n", view.label().c_str(), filename.c_str());

    for (size_t idim = 0; idim < ViewType::rank; ++idim) {
        const auto dim = view.extent(idim);
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }

    out.write(reinterpret_cast<const char*>(view_h.data()), sizeof(typename ViewType::value_type) * view.size());
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard{argc, argv};
    {
        using value_t = double;

        using function_t = reprospect::examples::kokkos::complex::ZPow4MinOne<value_t, Kokkos::CudaSpace>;
        using divisor_t = reprospect::examples::kokkos::complex::DivisorLogbScalbn<true>;
        using compute_t =
            reprospect::examples::kokkos::complex::ComputeColors<value_t, Kokkos::CudaSpace, divisor_t, function_t>;

        const Kokkos::Cuda exec{};

        const compute_t compute{exec, function_t{exec}, 1080, 1080, 150};
        compute.apply(exec);

        exec.fence();

        dump(compute.colors, std::filesystem::path(std::getenv("OUTPUT_DIR")) / "colors.bin");
        dump(compute.iterations, std::filesystem::path(std::getenv("OUTPUT_DIR")) / "iterations.bin");
    }
}
