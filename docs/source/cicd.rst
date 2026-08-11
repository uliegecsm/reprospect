CI/CD
=====

The CI/CD pipelines run the `ReProspect` tests and examples for a variety of configurations.

The current CI/CD strategy matrix covers:

- many CUDA versions
- many architectures
- ``nvcc`` or ``clang`` as CUDA compiler

.. literalinclude:: ../../.github/workflows/strategy.py
   :language: python
   :start-after: STRATEGY-MATRIX-START
   :end-before: STRATEGY-MATRIX-END

The CI/CD pipelines run the tests and examples in Docker containers.
For each entry in the strategy matrix, the CI/CD pipeline builds two images:

- the first image contains the dependencies needed to build and run the `ReProspect` tests for the given configuration;
- the second image contains, in addition, a Kokkos_ installation (currently |kokkos_sha|) and a compatible `Kokkos Tools`_ installation, as needed to run the `ReProspect` examples.

The full list of images the CI/CD pipelines build is as follows:

.. _images:

.. include:: generated/images.rst

These images are public and can be pulled anonymously from the GitHub Container Registry:

.. code-block:: bash

   docker pull ghcr.io/uliegecsm/reprospect/IMAGE_NAME
