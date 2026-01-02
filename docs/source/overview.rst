.. _overview:

Overview
========

`ReProspect` provides a framework for reproducible prospecting of CUDA applications.
It is implemented in `Python` and builds on the NVIDIA tools:
`Nsight Systems`_ for API tracing :cite:`nvidia-nsight-systems`,
`Nsight Compute`_ for kernel profiling :cite:`nvidia-nsight-compute`,
and the `CUDA binary utilities`_ for binary analysis :cite:`nvidia-cuda-binary-utilities`.

`ReProspect` provides an interface to launch these NVIDIA tools, as well as post-processing utilities to collect the results in tailored data structures.
It also provides assertion functions to verify that the collected data match expectations.

`ReProspect` allows the entire prospecting workflow to be encoded in a `Python` script,
thus making it reproducible, integrable into automated workflows, as well as easily shareable.
One of the objectives of `ReProspect` is to allow new use cases for CUDA code prospecting to be developed and explored,
such as the integration of such analyses in CI/CD pipelines and their increased use in collaborative code reviews to help motivate development decisions.

`ReProspect` supports three complementary forms of analysis:

   .. rubric:: API tracing

   Built on top of `Nsight Systems`_ (``nsys``) to record various runtime events, such as CUDA API calls, and examine application-level control flow.

   .. rubric:: Kernel profiling

   Built on top of `Nsight Compute`_ (``ncu``) to collect detailed per-kernel runtime metrics.

   .. rubric:: Binary analysis

   Built on top of the `CUDA binary utilities`_ (``cuobjdump`` and ``nvdisasm``) to extract information from CUDA binaries,
   such as the SASS instructions that make up a kernel, the associated control flow instructions, the resource usage, and so on.

In order to facilitate its integration in CI/CD pipelines, `ReProspect` has functionalities to interact with the build system,
such as functionalities to obtain information about the compiler toolchain. Currently, these functionalities have been implemented only for the `CMake`_ build system.

The main steps in a typical `ReProspect` case are illustrated in the following schematic (see also :ref:`analysis-workflow`):

.. tikz::
   :libs: arrows.meta, arrows, backgrounds, calc, fit, positioning, shapes
   :include: schematic.tikz

The `ReProspect` repository includes several examples relevant to the Kokkos_ :cite:`kokkos` performance portability library,
and in particular its CUDA backend. These examples demonstrate `ReProspect` capabilities
by examining several specific implementation details of Kokkos_.
