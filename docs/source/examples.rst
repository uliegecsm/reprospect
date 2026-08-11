.. _examples:

Examples
========

Running the examples
--------------------

The CI/CD pipelines build and run the examples in Docker containers for a variety of configurations.
The most convenient way to build and run the examples without reconstructing the dependencies is to reuse one of our images.

For example, for an ``x86_64`` platform, with a ``Blackwell120`` GPU, targeting a compiler toolchain using ``nvcc`` as the CUDA compiler and ``gcc`` as the host compiler,
the image ``cuda-gnu-14-nvidia-py3.13-kokkos-5.1.0:13.1.0-devel-ubuntu24.04-blackwell120`` may be used.

Start a container in interactive mode:

.. code-block:: bash

   docker run --rm -it --gpus=all --cap-add=SYS_ADMIN \
       ghcr.io/uliegecsm/reprospect/cuda-gnu-14-nvidia-py3.13-kokkos-5.1.0:13.1.0-devel-ubuntu24.04-blackwell120

Then, inside the container:

.. code-block:: bash

   git clone https://github.com/uliegecsm/reprospect.git /workspace/reprospect
   cd /workspace/reprospect

   cmake -S . --preset=gnu-nvidia \
       -DCMAKE_CUDA_ARCHITECTURES=120-real \
       -DReProspect_ENABLE_EXAMPLES=ON \
       -DReProspect_ENABLE_TESTS=OFF \
       -DReProspect_ENABLE_DOCS=OFF

   cmake --build --preset=gnu-nvidia
   ctest --preset=gnu-nvidia

For another configuration, choose the image from the :ref:`list of images the CI/CD pipelines build <images>`,
set ``CMAKE_CUDA_ARCHITECTURES`` to match the architecture of the GPU (e.g., ``70-real`` for ``Volta70``),
and select the ``preset`` matching the targeted compiler toolchain, namely, ``gnu-nvidia`` for ``nvcc`` as the CUDA compiler with ``gcc`` as the host compiler, ``clang-nvidia`` for ``nvcc`` as the CUDA compiler with ``clang`` as the host compiler, and ``clang`` for ``clang`` as the CUDA compiler.

Building and running the examples with your own compiler toolchain and Kokkos_ installation is also possible.
It requires:

- a CUDA Toolkit installation;
- `Nsight Systems`_ and `Nsight Compute`_ installations and the `CUDA binary utilities`_;
- the NVTX_ header ``nvtx3/nvtx3.hpp``;
- a `Google Benchmark`_ installation;
- a Kokkos_ installation with a compatible `Kokkos Tools`_ installation;
- CMake_;
- a C++20-capable toolchain.

If ``clang`` is the CUDA compiler, the compiler toolchain must also provide the ``llvm-cxxfilt`` demangler.
The CI/CD pipelines currently build and run the examples with Kokkos_ |kokkos_sha|.

Overview of the examples
------------------------

.. toctree::
   :maxdepth: 3

   examples/cuda
   examples/kokkos
