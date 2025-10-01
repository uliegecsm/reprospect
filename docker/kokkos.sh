set -ex

cmake_args=(
    "-DKokkos_ENABLE_SERIAL=ON"
    "-DKokkos_ENABLE_CUDA=ON"
    "-DKokkos_ARCH_${KOKKOS_ARCH}=ON"
    "-DCMAKE_INSTALL_PREFIX=/opt/kokkos-${KOKKOS_SHA}/${KOKKOS_CMAKE_PRESET}"
)

case "${KOKKOS_CMAKE_PRESET}" in
    gcc-nvcc)
        cmake_args+=(
            "-DCMAKE_CXX_COMPILER=g++"
            "-DCMAKE_CUDA_COMPILER=nvcc"
            "-DCMAKE_CUDA_HOST_COMPILER=g++"
            "-DCMAKE_CUDA_SEPARABLE_COMPILATION=ON"
        )
        ;;
    clang-nvcc)
        cmake_args+=(
            "-DCMAKE_CXX_COMPILER=clang++"
            "-DCMAKE_CUDA_COMPILER=nvcc"
            "-DCMAKE_CUDA_HOST_COMPILER=clang"
            "-DCMAKE_CUDA_SEPARABLE_COMPILATION=ON"
        )
        ;;
    clang)
        cmake_args+=(
            "-DCMAKE_CXX_COMPILER=clang++"
            "-DCMAKE_CUDA_COMPILER=clang++"
        )
        ;;
    *)
        echo "Error: Unknown CMake preset '${KOKKOS_CMAKE_PRESET}'."
        exit -1
        ;;
esac

cmake -S . -B build "${cmake_args[@]}"

cmake --build build --target=install -j4
