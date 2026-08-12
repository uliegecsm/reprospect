Installing `ReProspect`
=======================

`ReProspect` can be installed *via* ``pip`` as follows:

.. code-block:: bash

   pip install reprospect

Verification
------------

To ensure that `ReProspect` was installed correctly, the installation can be verified with a few lines of Python.
Here, we use the :py:class:`~reprospect.utils.detect.GPUDetector` class from the :py:mod:`reprospect.utils.detect` module to detect the visible GPUs on the system.

.. code-block:: python

   from reprospect.utils.detect import GPUDetector
   print(GPUDetector().detect())

The output should be similar to:

.. code-block:: text

                                          uuid  index                        name compute_cap  architecture
   0  GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx      0  NVIDIA GeForce RTX 5070 Ti        12.0  BLACKWELL120

The column values identify the specific GPUs visible on the system and will differ on your system.
The output is empty if no NVIDIA GPUs are visible on the system.

Optional dependency on ``cuda-bindings``
----------------------------------------

Some modules of `ReProspect` such as :py:mod:`reprospect.tools.device_properties` depend
on the `Low-level Python Bindings for CUDA`_ (``cuda-bindings``) package.

It is recommended to install the version of ``cuda-bindings`` that matches your installed CUDA driver.

`ReProspect` **does not install** this dependency automatically.
Attempting to let `ReProspect` do so would (1) overcomplicate the installation process and make it
non-standard for a Python package, and (2) might not reliably install the correct version of
``cuda-bindings`` for your environment.

For recent CUDA versions, ``cuda-bindings`` can be installed via ``pip``:

.. code-block:: bash

   pip install cuda-bindings==<cuda-version>

For instance, for CUDA 13.0.1, you may use ``pip install cuda-bindings==13.0.1``.
If an exact match is not available, install the closest version with the same major and minor version numbers.
For instance, for CUDA 12.8.1, install version 12.8.0.

For older CUDA versions, ``cuda-bindings`` was part of the ``cuda-python`` package, which can
be installed likewise via ``pip``.

Note that `ReProspect` provides a helper script, :py:mod:`reprospect.utils.installers.cuda_bindings`,
which can be used or whose logic can be followed to conveniently install the appropriate package
with the appropriate version.
