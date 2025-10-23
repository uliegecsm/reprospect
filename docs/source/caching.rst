Caching
=======

Profiling runs with ``ncu`` and ``nsys`` are time-consuming.
Since developers are encouraged to write profiling tests whose results remain stable across runs,
re-profiling an executable on every CI pipeline execution, event when this executable and its dependencies
have not changed, may be unnecessary.
Additionally, ``ncu`` and ``nsys`` often require exclusive resource access, which can serialize tests
and further slow down CI pipelines.

To address this issue, we propose a file-based caching system modeled after ``ccache``.
The cache key is derived from a hash of the environment, the executable, its library dependencies and the command-line arguments,
thus allowing results to be retrieved from cache when conditions match.

We recommend writing tests that may depend on GPU architecture and device characteristics,
but remain independent of machine-specific factors such as host CPU or PCIe topology.

.. note::

    Sharing caches across runners is not recommended, as profiling results may vary depending on machine configuration.
    The proposed setup targets scenarios where each CI runner executes jobs independently and maintains its own persistent directory for caching.
    For example, jobs might run inside a `Docker` container on a GitHub CI, and store their cache in a mounted ``$HOME`` directory.

- The cacher for ``ncu`` is :py:class:`reprospect.tools.ncu.Cacher`.
- The cacher for ``nsys`` is :py:class:`reprospect.tools.nsys.Cacher`.
