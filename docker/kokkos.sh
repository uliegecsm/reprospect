set -ex

# Kokkos
cd /opt/kokkos-sources/kokkos-${KOKKOS_SHA}

# When nvcc is used as CUDA compiler, use 'nvcc_wrapper'.
if [[ "${KOKKOS_CUDA_COMPILER}" == "nvcc" ]];then
    cmake_args_kokkos=(
        "-DCMAKE_CXX_COMPILER=/opt/kokkos-sources/kokkos-${KOKKOS_SHA}/bin/nvcc_wrapper"
    )
    export NVCC_WRAPPER_DEFAULT_COMPILER=${KOKKOS_CXX_COMPILER}
else
    cmake_args_kokkos=(
        "-DCMAKE_CXX_COMPILER=${KOKKOS_CXX_COMPILER}"
        "-DCMAKE_CUDA_COMPILER=${KOKKOS_CUDA_COMPILER}"
    )
fi

cmake_args_kokkos+=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_STANDARD=20"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_CXX_EXTENSIONS=OFF"
    "-DKokkos_ENABLE_SERIAL=ON"
    "-DKokkos_ENABLE_CUDA=ON"
    "-DKokkos_ARCH_${KOKKOS_ARCH}=ON"
    "-DCMAKE_INSTALL_PREFIX=/opt/kokkos-${KOKKOS_SHA}/${KOKKOS_INSTALL_SUFFIX}"
)

cmake -S . -B build "${cmake_args_kokkos[@]}"

cmake --build build --target=install -j4

# Kokkos Tools
cd /opt/kokkos-tools-sources/kokkos-tools-${KOKKOS_TOOLS_SHA}

cmake_args_kokkos_tools=(
    "-DCMAKE_CXX_COMPILER=${KOKKOS_CXX_COMPILER}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_STANDARD=20"
    "-DKokkosTools_ENABLE_SINGLE=OFF"
    "-DKokkosTools_ENABLE_MPI=OFF"
    "-DCMAKE_INSTALL_PREFIX=/opt/kokkos-tools-${KOKKOS_TOOLS_SHA}/${KOKKOS_TOOLS_INSTALL_SUFFIX}"
)

Kokkos_ROOT=/opt/kokkos-${KOKKOS_SHA}/${KOKKOS_INSTALL_SUFFIX} \
    cmake -S . -B build "${cmake_args_kokkos_tools[@]}"

cmake --build build --target=install -j4
