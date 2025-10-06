set -ex

cmake_args=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_STANDARD=20"
    "-DBUILD_SHARED_LIBS=ON"
    "-DKokkosTools_ENABLE_SINGLE=ON"
    "-DKokkosTools_ENABLE_MPI=OFF"
    "-DCMAKE_INSTALL_PREFIX=/opt/kokkos-tools-${KOKKOS_TOOLS_SHA}/${KOKKOS_TOOLS_INSTALL_SUFFIX}"
)

# When nvcc is used as CUDA compiler, use 'nvcc_wrapper'.
if [[ "${KOKKOS_CUDA_COMPILER}" == "nvcc" ]];then
    cmake_args+=(
        "-DCMAKE_CXX_COMPILER=$Kokkos_ROOT/bin/nvcc_wrapper"
    )
    export NVCC_WRAPPER_DEFAULT_COMPILER=${KOKKOS_CXX_COMPILER}
else
    cmake_args+=(
        "-DCMAKE_CXX_COMPILER=${KOKKOS_CXX_COMPILER}"
    )
fi

cmake -S . -B build "${cmake_args[@]}"

cmake --build build --target=install -j4
