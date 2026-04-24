.. _analysis-workflow:

Analysis workflow
=================

Three components
----------------

.. _api_tracing:

API tracing
~~~~~~~~~~~

* :py:class:`reprospect.tools.nsys.Cacher`
* :py:class:`reprospect.tools.nsys.Session`
* :py:class:`reprospect.tools.nsys.Report`

.. _kernel_profiling:

Kernel profiling
~~~~~~~~~~~~~~~~

* :py:class:`reprospect.tools.ncu.Cacher`
* :py:class:`reprospect.tools.ncu.Session`
* :py:class:`reprospect.tools.ncu.Report`

.. _binary_analysis:

Binary analysis
~~~~~~~~~~~~~~~

* :py:class:`reprospect.tools.binaries.CuObjDump`
* :py:class:`reprospect.tools.binaries.ELF`
* :py:class:`reprospect.tools.binaries.NVDisasm`
* :py:class:`reprospect.tools.sass.Decoder`
* :py:class:`reprospect.test.sass.instruction.instruction.InstructionMatcher`
* :py:class:`reprospect.test.sass.composite_impl.SequenceMatcher`

From sources to binaries
------------------------

* :py:class:`reprospect.utils.cmake.FileAPI`
