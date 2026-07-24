.. _analysis-workflow:

Analysis workflow
=================

Three components
----------------

.. _api_tracing:

API tracing
~~~~~~~~~~~

* :py:class:`reprospect.tools.nsys.cacher.Cacher`
* :py:class:`reprospect.tools.nsys.session.Session`
* :py:class:`reprospect.tools.nsys.report.Report`

.. _kernel_profiling:

Kernel profiling
~~~~~~~~~~~~~~~~

* :py:class:`reprospect.tools.ncu.cacher.Cacher`
* :py:class:`reprospect.tools.ncu.session.Session`
* :py:class:`reprospect.tools.ncu.report.Report`

.. _binary_analysis:

Binary analysis
~~~~~~~~~~~~~~~

* :py:class:`reprospect.testing.binaries.sass.instruction.instruction.InstructionMatcher`
* :py:class:`reprospect.testing.binaries.sass.sequence.sequence.SequenceMatcher`
* :py:class:`reprospect.tools.binaries.cuobjdump.CuObjDump`
* :py:class:`reprospect.tools.binaries.elf.ELF`
* :py:class:`reprospect.tools.binaries.nvdisasm.NVDisasm`
* :py:class:`reprospect.tools.binaries.sass.decoder.Decoder`

From sources to binaries
------------------------

* :py:class:`reprospect.utils.cmake.FileAPI`
