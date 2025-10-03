set -ex

cmake_args=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_STANDARD=20"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_CXX_EXTENSIONS=OFF"
    "-DKokkos_ENABLE_SERIAL=ON"
    "-DKokkos_ENABLE_CUDA=ON"
    "-DKokkos_ARCH_${KOKKOS_ARCH}=ON"
    "-DCMAKE_INSTALL_PREFIX=/opt/kokkos-${KOKKOS_SHA}/${KOKKOS_INSTALL_SUFFIX}"
)

# When nvcc is used as CUDA compiler, use 'nvcc_wrapper'.
if [[ "${KOKKOS_CUDA_COMPILER}" == "nvcc" ]];then
    cmake_args+=(
        "-DCMAKE_CXX_COMPILER=$PWD/bin/nvcc_wrapper"
    )
    export NVCC_WRAPPER_DEFAULT_COMPILER=${KOKKOS_CXX_COMPILER}
else
    cmake_args+=(
        "-DCMAKE_CXX_COMPILER=${KOKKOS_CXX_COMPILER}"
        "-DCMAKE_CUDA_COMPILER=${KOKKOS_CUDA_COMPILER}"
    )
fi

cmake -S . -B build "${cmake_args[@]}"

cmake --build build --target=install -j4
