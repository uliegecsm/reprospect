Installing `ReProspect`
=======================

Some `ReProspect` modules depend on `Low-level Python Bindings for CUDA`_ (``cuda-bindings``).
It is recommended to install the version of ``cuda-bindings`` that matches the installed CUDA driver.

`ReProspect` **does not install** this dependency automatically.
Attempting do so would (1) overcomplicate the installation process and make it non-standard for a Python package,
and (2) might not reliably install the correct version of ``cuda-bindings`` for your environment.

Instead, it provides a helper script, :py:mod:`reprospect.installers.cuda_bindings`,
which can be used to conveniently install the appropriate version.
