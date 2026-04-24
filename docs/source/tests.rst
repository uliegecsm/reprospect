Tests
=====

The test directory structure mirrors
the :py:mod:`reprospect` package source directory structure.

The tests for the three main subpackages can be found in:

.. toctree::
   :maxdepth: 1

   tests/test
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

CI/CD strategy matrix
---------------------

The CI/CD strategy matrix covers:

- many CUDA versions
- many architectures
- ``nvcc`` or ``clang`` as CUDA compiler

.. literalinclude:: ../../.github/workflows/strategy.py
   :language: python
   :start-after: STRATEGY-MATRIX-START
   :end-before: STRATEGY-MATRIX-END
