Developing in `ReProspect`
==========================

Compiling `Rust` extension modules
----------------------------------

`ReProspect` optimizes performance-critical code paths
using `Rust` via `PyO3 <https://github.com/PyO3/pyo3>`_.
The strategy is to implement hot paths as `Rust` extension modules
while minimizing the overhead of crossing the Python-`Rust` boundary (*e.g.* by batching operations).

Building extensions during development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compile the `Rust` extension modules and install them in-place (so they can be
imported directly from your `Git` source tree):

.. code-block:: bash

    python setup.py build_rust --inplace --release
