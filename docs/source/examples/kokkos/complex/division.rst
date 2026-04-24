Division
========

Dividing complex numbers is more subtle than it appears. The naïve formula:

.. math::

   \frac{a + bi}{c + di} = \frac{ac + bd + (bc - ad)\,i}{c^2 + d^2}

computes :math:`c^2 + d^2`, which can overflow for large inputs. Underflow regimes exist as well.
Many algorithms have therefore been proposed to make complex division more robust
:cite:`baudin-2012-robust-complex-division-scilab`, each with different trade-offs in
numerical accuracy and performance.
These algorithms typically use a scaling strategy.
A further challenge in complex division is the handling of edge cases such as infinite or zero operands.

The standard **ISO/IEC 9899 (Annex G)** :cite:`iso-iec-9899-2024` raises the issue of overflow
and underflow and specifies the expected outcomes for the edge cases to comply with **IEC 60599** real floating-point arithmetic.
It also suggests an implementation that adopts a scaling approach using the :code:`scalbn` function
and relies on branching to handle the edge cases.
Compilers such as `LLVM <https://github.com/llvm/llvm-project/blob/2342db00ab4d0305580814fb00f477b4b5cebec6/compiler-rt/lib/builtins/divdc3.c>`_
and libraries such as `CCCL <https://github.com/NVIDIA/cccl/blob/a91db6e2a022a7aa03b37873f0d4caf5ac81281d/libcudacxx/include/cuda/std/__complex/complex.h#L469>`_
follow this suggested implementation.
They also expose mechanisms to disable the branching,
which can save computation cycles when such operands are not expected.

The current implementation of `complex division in Kokkos <https://github.com/kokkos/kokkos/blob/e3a19e1c5bd4dcc5f59fba216b72967692f5a33e/core/src/Kokkos_Complex.hpp#L181-L206>`_
does not follow the **ISO/IEC 9899 (Annex G)** implementation.
Instead, it adopts a scaling approach using a division, and it only partially
complies with the handling of the edge cases.
Their implementation thus entails more divisions than the **ISO/IEC 9899 (Annex G)** implementation, see
:py:meth:`examples.kokkos.complex.example_division.TestSASS.test_norm_division_uses_more_rcp`.

This example also explores a potential improvement of the
**ISO/IEC 9899 (Annex G)** implementation that replaces the use of :code:`logb` followed by a cast to :code:`int`
with a use of :code:`ilogb`.
This change decreases the number of `fp64` instructions and cycles, see
:py:meth:`examples.kokkos.complex.example_division.TestNCU.test_fp64_instructions_and_cycles`.

This example thus compares three implementations:

1. :py:attr:`examples.kokkos.complex.example_division.Method.NormDivision`
2. :py:attr:`examples.kokkos.complex.example_division.Method.LogbScalbn`
3. :py:attr:`examples.kokkos.complex.example_division.Method.ILogbScalbn`

Both :py:attr:`examples.kokkos.complex.example_division.Method.LogbScalbn`
and :py:attr:`examples.kokkos.complex.example_division.Method.ILogbScalbn` use a common C++ implementation templated
on a flag that allows the handling of edge cases to be disabled
and another flag that controls the use of :code:`ilogb`.

This example has three parts:

1. A :download:`compliance test <../../../../../examples/kokkos/complex/example_division_compliance.cpp>` that checks
   whether each implementation complies with the expected outcomes for the edge cases.
2. A binary analysis in :py:meth:`examples.kokkos.complex.example_division.TestSASS`
   and a kernel profiling in :py:meth:`examples.kokkos.complex.example_division.TestNCU`.
3. A benchmark in :py:meth:`examples.kokkos.complex.example_division_benchmarking.TestDivision`.

The benchmark compares the performance of the three implementations using the Newton fractal
:cite:`wikipedia-newton-fractal`, which requires one complex division
per iteration per pixel:

.. figure:: ../../../../../artifacts/examples/kokkos/complex/example_division_plot.cpp/fractal.svg

    Newton fractal for :math:`z^4 - 1` with relaxation factor :math:`a = 1`.

.. automodule:: examples.kokkos.complex.example_division
.. automodule:: examples.kokkos.complex.example_division_benchmarking
.. automodule:: examples.kokkos.complex.example_division_plot
