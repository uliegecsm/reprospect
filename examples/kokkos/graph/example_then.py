import re
import sys

from reprospect.test import CMakeAwareTestCase
from reprospect.tools import binaries
from reprospect.tools.binaries.elf import NvInfoEIATTR

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class TestThen(CMakeAwareTestCase):
    """
    Analyze the :code:`then` node of :code:`Kokkos::Experimental::Graph`.

    It uses :file:`examples/kokkos/graph/example_then.cpp`.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_graph_then'

class TestELF(TestThen):
    @property
    def signature_matcher(self) -> re.Pattern[str]:
        """
        Expected Kokkos driver signature matcher.

        The signature indicates that :code:`Kokkos::LaunchBounds<1, 0>` is used.

        References:

        * https://github.com/kokkos/kokkos/blob/94dd7e0ee93d30bffbf7255d36bf74e75fe801bf/core/src/impl/Kokkos_GraphNodeThenImpl.hpp#L35-L41
        * https://github.com/kokkos/kokkos/blob/94dd7e0ee93d30bffbf7255d36bf74e75fe801bf/core/src/Cuda/Kokkos_Cuda_KernelLaunch.hpp#L101-L108
        """
        zero = r'(\(unsigned int\)0|0u)'
        one  = r'(\(unsigned int\)1|1u)'
        arg  = r'[A-Za-z0-9 :<>_,]+'
        return re.compile(
            r'void Kokkos::Impl::cuda_parallel_launch_local_memory<'
            r'Kokkos::Impl::ParallelFor<Kokkos::Impl::ThenWrapper<'
            r'reprospect::examples::kokkos::graph::Functor<'
            r'Kokkos::View<int, Kokkos::CudaUVMSpace>>>, '
            rf'Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::Impl::IsGraphKernelTag, Kokkos::LaunchBounds<{one}, {zero}>, void>, '
            rf'Kokkos::Cuda>, {one}, {zero}>\({arg}\)',
        )

    def test_max_threads_1_1_1(self) -> None:
        """
        Validate that the CUDA kernel whose signature matches :py:meth:`signature_matcher` has its
        :py:const:`reprospect.tools.binaries.elf.NvInfoEIATTR.MAX_THREADS` attribute set to `(1, 1, 1)`, indicating that
        :code:`__launch_bounds__(1, ...)` was used in the source code.

        References:

        * https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-bounds
        """
        cuobjdump, cubin = binaries.CuObjDump.extract(
            file = self.executable,
            arch = self.arch,
            cwd = self.cwd,
            sass = False,
            demangler = None,
            cubin = f'{self.executable.name}.1.{self.arch.as_sm}.cubin',
        )
        assert cubin.is_file()

        mask = (
            cuobjdump.symtab['name'].str.contains('Kokkos') &
            cuobjdump.symtab['name'].str.contains('cuda_parallel_launch_local_memory') &
            cuobjdump.symtab['name'].str.contains('IsGraphKernelTag')
        )
        filtered = cuobjdump.symtab[mask]
        demangled = filtered['name'].apply(self.demangler.demangle)
        [mangled] = filtered.loc[demangled.str.match(self.signature_matcher), 'name']

        with binaries.ELF(file = cubin) as elf:
            nvinfo = elf.nvinfo(mangled = mangled)
            [max_threads] = nvinfo.iter(eiattr = NvInfoEIATTR.MAX_THREADS)

            assert max_threads.value == (1, 1, 1)
