.. _analysis-workflow:

Analysis workflow |:test_tube:|
===============================

Analysis is 3-fold |:mag:|
--------------------------

Binary analysis
~~~~~~~~~~~~~~~

* :py:class:`reprospect.tools.binaries.CuObjDump`
* :py:class:`reprospect.tools.binaries.NVDisasm`
* :py:class:`reprospect.tools.sass.Decoder`

Kernel profiling |:female_detective:|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:class:`reprospect.tools.ncu.Cacher`
* :py:class:`reprospect.tools.ncu.Session`
* :py:class:`reprospect.tools.ncu.Report`

API tracing |:movie_camera:|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:class:`reprospect.tools.nsys.Cacher`
* :py:class:`reprospect.tools.nsys.Session`
* :py:class:`reprospect.tools.nsys.Report`

See for example:

* :py:class:`examples.kokkos.view.example_allocation_tracing.TestNSYS`

From sources to binaries |:bricks:|
-----------------------------------

* :py:class:`reprospect.utils.cmake.FileAPI`

