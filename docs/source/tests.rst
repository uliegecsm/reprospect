Tests
=====

Running the tests
-----------------

The CI/CD pipelines build and run the tests in Docker containers for a variety of configurations.
The most convenient way to build and run the tests without reconstructing the dependencies is to reuse one of our images.

For example, for an ``x86_64`` platform, with a ``Blackwell120`` GPU, targeting a compiler toolchain using ``nvcc`` as the CUDA compiler and ``gcc`` as the host compiler,
the image ``cuda-gnu-14-nvidia-py3.13:13.1.0-devel-ubuntu24.04`` may be used.

Start a container in interactive mode:

.. code-block:: bash

   docker run --rm -it --gpus=all --cap-add=SYS_ADMIN \
       ghcr.io/uliegecsm/reprospect/cuda-gnu-14-nvidia-py3.13:13.1.0-devel-ubuntu24.04

The GPU architecture enters through ``CMAKE_CUDA_ARCHITECTURES`` below.
Only the images containing a Kokkos_ installation used for the `ReProspect` examples are specific to a GPU architecture.

Then, inside the container:

.. code-block:: bash

   git clone https://github.com/uliegecsm/reprospect.git /workspace/reprospect
   cd /workspace/reprospect

   cmake -S . --preset=gnu-nvidia \
       -DCMAKE_CUDA_ARCHITECTURES=120-real \
       -DReProspect_ENABLE_EXAMPLES=OFF \
       -DReProspect_ENABLE_TESTS=ON \
       -DReProspect_ENABLE_DOCS=OFF

   cmake --build --preset=gnu-nvidia
   ctest --preset=gnu-nvidia

For another configuration, choose the image from the :ref:`list of images the CI/CD pipelines build <images>`,
set ``CMAKE_CUDA_ARCHITECTURES`` to match the architecture of the GPU (e.g., ``70-real`` for ``Volta70``),
and select the ``preset`` matching the targeted compiler toolchain, namely, ``gnu-nvidia`` for ``nvcc`` as the CUDA compiler with ``gcc`` as the host compiler, ``clang-nvidia`` for ``nvcc`` as the CUDA compiler with ``clang`` as the host compiler, and ``clang`` for ``clang`` as the CUDA compiler.

Overview of the tests
---------------------

The test directory structure mirrors
the :py:mod:`reprospect` package source directory structure.

The tests for the three main subpackages can be found in:

.. toctree::
   :maxdepth: 1

   tests/testing
   tests/tools
   tests/utils

Tests that involve many subpackages are grouped in:

.. toctree::
   :maxdepth: 1

   tests/integration

Utilities for testing
---------------------

`ReProspect` tests use helpers for diverse tasks such as
compiling or
extracting random bits from cuBLAS.

.. toctree::
   :maxdepth: 1

   tests/compilation
   tests/cublas
   tests/parameters

Test assets
-----------

Test assets are located in:

* a subdirectory named `assets` next to the test file that uses them
* the central `tests/assets` directory if shared across multiple tests

.. toctree::
   :maxdepth: 1

   tests/assets
