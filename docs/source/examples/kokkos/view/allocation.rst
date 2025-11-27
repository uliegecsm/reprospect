`Kokkos::View` allocation
=========================

The use of API tracing to elucidate benchmarking results
--------------------------------------------------------

A benchmark comparing :code:`Kokkos` with native CUDA can be found in :py:mod:`examples.kokkos.view.example_allocation_benchmarking`.

To better understand the results, a comprehensive CUDA API tracing test is required.
Indeed, as tested in :py:class:`examples.kokkos.view.example_allocation_tracing.TestNSYS`,
the code path followed by :code:`Kokkos` depends on the memory space in which the allocation happens
as well as its size.
The tracing also brings to light that :code:`Kokkos` always copies the :code:`Kokkos::View` shared allocation
header, which greatly impacts the performance compared to allocating manually with CUDA (see also https://github.com/kokkos/kokkos/pull/8440).

Benchmarking
------------

.. figure:: example_allocation_benchmarking.svg

    Comparison of repeated buffer allocation/deallocation using :code:`Kokkos` or native CUDA, with stream-ordered allocation or not.
    CUDA 13.0.0, :code:`Kokkos` 4.7.01, NVIDIA GeForce RTX 5070 Ti, :lastcommit:`docs/source/examples/kokkos/view/example_allocation_benchmarking.svg`.
    Note that the results may vary with machine setup.

.. automodule:: examples.kokkos.view.example_allocation_benchmarking

Tracing
-------

.. automodule:: examples.kokkos.view.example_allocation_tracing
