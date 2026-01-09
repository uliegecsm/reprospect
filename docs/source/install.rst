Installing `ReProspect`
=======================

`ReProspect` can be installed *via* ``pip`` as follows:

.. code-block:: bash

   pip install reprospect

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

Note that `ReProspect` provides a helper script, :py:mod:`reprospect.installers.cuda_bindings`,
which can be used or whose logic can be followed to conveniently install the appropriate package
with the appropriate version.
