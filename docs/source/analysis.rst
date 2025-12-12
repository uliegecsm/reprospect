.. _analysis-workflow:

Analysis workflow
=================

Three components
----------------

API tracing
~~~~~~~~~~~

* :py:class:`reprospect.tools.nsys.Cacher`
* :py:class:`reprospect.tools.nsys.Session`
* :py:class:`reprospect.tools.nsys.Report`

Kernel profiling
~~~~~~~~~~~~~~~~

* :py:class:`reprospect.tools.ncu.Cacher`
* :py:class:`reprospect.tools.ncu.Session`
* :py:class:`reprospect.tools.ncu.Report`

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

