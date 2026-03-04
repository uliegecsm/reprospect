Division
========

Dividing complex numbers is more subtle than it appears. The naïve formula:

.. math::

   \frac{a + bi}{c + di} = \frac{ac + bd + (bc - ad)\,i}{c^2 + d^2}

computes :math:`c^2 + d^2`, which can overflow for large inputs. Underflow regimes exist as well.
Many algorithms have therefore been proposed to make complex division more robust
:cite:`baudin-2012-robust-complex-division-scilab`, each with different trade-offs in
numerical accuracy and performance.

Some implementations also claim **ISO/IEC 60559** compliance :cite:`iso-iec-9899-2024`,
which mandates specific handling of edge cases such as infinite or zero operands.
The :download:`compliance test <../../../../../examples/kokkos/complex/example_division_compliance.cpp>`
can be used to validate several implementations against these requirements.
Note that libraries such as `CCCL <https://github.com/NVIDIA/cccl/blob/a91db6e2a022a7aa03b37873f0d4caf5ac81281d/libcudacxx/include/cuda/std/__complex/complex.h#L469>`_
expose an opt-out mechanism for this handling, which can save computation cycles when such operands are not expected.

This example focuses on performance using the Newton fractal
:cite:`wikipedia-newton-fractal` as a benchmark, which requires one complex division
per iteration per pixel.

.. figure:: ../../../../../artifacts/examples/kokkos/complex/example_division_plot.cpp/fractal.svg

    Newton fractal for :math:`z^4 - 1` with relaxation factor :math:`a = 1`.

.. automodule:: examples.kokkos.complex.example_division
.. automodule:: examples.kokkos.complex.example_division_benchmarking
.. automodule:: examples.kokkos.complex.example_division_plot
