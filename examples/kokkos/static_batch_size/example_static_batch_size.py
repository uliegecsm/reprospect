"""
`PR #8164 <https://github.com/kokkos/kokkos/pull/8164>`_
introduced :code:`Kokkos::Experimental::StaticBatchSize`.
Though the default value of 1 was not intended to impact the generated code,
it unintentionally increased the register usage.
This performance regression was not caught by any performance test or reviewer,
and was only corrected in `PR #9123 <https://github.com/kokkos/kokkos/pull/9123>`_.

This example illustrates how `ReProspect` could have been used
alongside `PR #8164 <https://github.com/kokkos/kokkos/pull/8164>`_
to assert that a unit batch size leaves register usage unchanged,
catching the regression before merging.
Such a test is demonstrated in
:py:meth:`examples.kokkos.static_batch_size.example_static_batch_size.TestStaticBatchSize.test_detailed_register_usage`.
"""


import pathlib
import re
import sys
import typing

import pytest

from reprospect.testing import CMakeAwareTestCase
from reprospect.tools.binaries import CuObjDump, DetailedRegisterUsage, NVDisasm
from reprospect.tools.sass.decode import RegisterType

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class TestStaticBatchSize(CMakeAwareTestCase):
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_static_batch_size_static_batch_size'

    BATCH_SIZES: typing.Final[tuple[int, ...]] = (1, 2)

    @staticmethod
    def signature_matcher(*, batch_size: int) -> re.Pattern[str]:
        batch_size_str = rf'(\(unsigned int\){batch_size}|{batch_size}u)'
        return re.compile(
            r'void Kokkos::Impl::cuda_parallel_launch_local_memory<'
            r'Kokkos::Impl::ParallelFor<'
            r'reprospect::examples::kokkos::static_batch_size::Increment<'
            r'Kokkos::View<unsigned short, Kokkos::CudaUVMSpace>>, '
            rf'Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::Experimental::StaticBatchSize<{batch_size_str}>>, '
            r'Kokkos::Cuda>>',
        )

    @property
    def cubin(self) -> pathlib.Path:
        return self.cwd / f'{self.get_target_name()}.1.{self.arch.as_sm}.cubin'

    @pytest.fixture(scope='class')
    def cuobjdump(self) -> CuObjDump:
        return CuObjDump.extract(
            file=self.executable,
            arch=self.arch, sass=True,
            cwd=self.cwd, cubin=self.cubin.name,
            demangler=self.demangler,
        )[0]

    @pytest.fixture(scope='class')
    def nvdisasm(self, cuobjdump: CuObjDump) -> NVDisasm:
        return NVDisasm(file=cuobjdump.file, arch=self.arch)

    @pytest.fixture(scope='class')
    def detailed_register_usage(self, cuobjdump: CuObjDump, nvdisasm: NVDisasm) -> dict[int, DetailedRegisterUsage]:
        def get_registers(batch_size: int) -> DetailedRegisterUsage:
            pattern = self.signature_matcher(batch_size=batch_size)
            [demangled] = (sig for sig in cuobjdump.functions if pattern.search(sig) is not None)
            mangled = cuobjdump.functions[demangled].symbol
            nvdisasm.extract_register_usage_from_liveness_range_info(mangled=(mangled,))
            assert (registers := nvdisasm.functions[mangled].registers) is not None
            return registers
        return {batch_size: get_registers(batch_size) for batch_size in self.BATCH_SIZES}

    def test_detailed_register_usage(self, detailed_register_usage: dict[int, DetailedRegisterUsage]) -> None:
        """
        Check detailed register usage for each batch size.

        .. note::

            Once https://github.com/kokkos/kokkos/pull/9123 is merged, the :py:attr:`reprospect.tools.sass.decode.RegisterType.GPR` count
            will decrease for the unit batch size case.
        """
        match self.arch.compute_capability:
            case 70:
                expt_stb_1 = {RegisterType.GPR: (10, 10), RegisterType.PRED: (2, 2)}
                expt_stb_2 = {RegisterType.GPR: (16, 12), RegisterType.PRED: (2, 2)}
            case 75:
                expt_stb_1 = {RegisterType.GPR: (10, 8), RegisterType.PRED: (2, 2), RegisterType.UGPR: (6, 2)}
                expt_stb_2 = {RegisterType.GPR: (10, 8), RegisterType.PRED: (2, 2), RegisterType.UGPR: (8, 4)}
            case 80 | 86 | 89:
                expt_stb_1 = {RegisterType.GPR: (10, 8), RegisterType.PRED: (2, 2), RegisterType.UGPR: (8, 4)}
                expt_stb_2 = {RegisterType.GPR: (10, 8), RegisterType.PRED: (2, 2), RegisterType.UGPR: (10, 6)}
            case 90:
                match self.compiler(toolchain='CUDA').id:
                    case 'NVIDIA':
                        expt_stb_1 = {RegisterType.GPR: (12, 9), RegisterType.PRED: (2, 2), RegisterType.UGPR: (10, 6)}
                        expt_stb_2 = {RegisterType.GPR: (12, 9), RegisterType.PRED: (2, 2), RegisterType.UGPR: (12, 8)}
                    case 'Clang':
                        expt_stb_1 = {RegisterType.GPR: (12, 10), RegisterType.PRED: (1, 1), RegisterType.UGPR: (9, 5)}
                        expt_stb_2 = {RegisterType.GPR: (14, 10), RegisterType.PRED: (1, 1), RegisterType.UGPR: (10, 6)}
                    case _:
                        raise ValueError
            case 100 | 103 | 120 | 121:
                match self.compiler(toolchain='CUDA').id:
                    case 'NVIDIA':
                        expt_stb_1 = {RegisterType.GPR: (14, 10), RegisterType.PRED: (1, 1), RegisterType.UGPR: (10, 6)}
                        expt_stb_2 = expt_stb_1
                    case 'Clang':
                        expt_stb_1 = {RegisterType.GPR: (12, 10), RegisterType.PRED: (1, 1), RegisterType.UGPR: (10, 6)}
                        expt_stb_2 = {RegisterType.GPR: (14, 10), RegisterType.PRED: (1, 1), RegisterType.UGPR: (10, 6)}
                    case _:
                        raise ValueError
            case _:
                raise ValueError(self.arch)

        assert detailed_register_usage[1] == expt_stb_1
        assert detailed_register_usage[2] == expt_stb_2
