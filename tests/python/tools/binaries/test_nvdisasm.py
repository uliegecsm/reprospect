import pathlib
import typing
import unittest

import pytest

from reprospect.tools.architecture      import NVIDIAArch
from reprospect.tools.binaries          import CuObjDump, CuppFilt, LlvmCppFilt, ResourceType, NVDisasm
from reprospect.tools.binaries.nvdisasm import Function
from reprospect.tools.sass.decode       import RegisterType
from reprospect.utils                   import cmake

from tests.python.compilation import get_compilation_output, get_cubin_name
from tests.python.parameters  import Parameters, PARAMETERS

class TestFunction:
    """
    Tests related to :py:class:`reprospect.tools.binaries.nvdisasm.Function`.
    """
    REGISTERS : typing.Final[dict[RegisterType, tuple[int, int]]] = {
        RegisterType.GPR  : (8, 7),
        RegisterType.PRED : (1, 1),
        RegisterType.UGPR : (7, 3),
    }

    def test_string_representation(self) -> None:
        """
        Test string representation of :py:class:`reprospect.tools.binaries.nvdisasm.Function`.
        """
        assert str(Function(registers = self.REGISTERS)) == """\
┏━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃      ┃ Span    ┃ Used ┃
┡━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ GPR  │ R0-R7   │ 7    │
│ PRED │ P0      │ 1    │
│ UGPR │ UR0-UR6 │ 3    │
└──────┴─────────┴──────┘
"""

class TestNVDisasm:
    """
    Tests related to :py:class:`reprospect.tools.binaries.NVDisasm`.
    """
    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.

        .. note:

             ``cuobjdump`` reports that the kernel uses 10 registers.
            However, ``nvdisasm`` reports that the kernel uses the span ``R0-R7``, hence a total of 8 registers.
            The reason why ``cuobjdump`` reports 2 more registers than ``nvdisasm``
            is not clear, but it has been observed elsewhere too.

            See for instance:

            * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/UserGuide.md?plain=1#L262
        """
        CUDA_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cu'
        SYMBOL    : typing.Final[str] = '_Z12saxpy_kernelfPKfPfj'
        SIGNATURE : typing.Final[str] = CuppFilt.demangle(SYMBOL)

        SASS_ANNOTATED_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'saxpy.sass.annotated'

        @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
        def test_from_object(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CUDA_FILE` as object, extract cubin and run ``nvdisasm``.
            """
            output, _ = get_compilation_output(
                source = self.CUDA_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = True,
                cmake_file_api = cmake_file_api,
            )

            cuobjdump, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = output.stem,
            )

            # cuobjdump reports a register usage of 10 GPRs.
            assert cuobjdump.functions[self.SIGNATURE].ru[ResourceType.REGISTER] == 10

            # nvdisasm indicates that the span R0-R7 of GPR registers is used.
            disasm = NVDisasm(file = cubin, arch = parameters.arch)
            disasm.extract_register_usage_from_liveness_range_info(mangled = (self.SYMBOL,))

            assert len(disasm.functions) == 1
            assert self.SYMBOL in disasm.functions

            match parameters.arch.compute_capability.as_int:
                case 70 | 75:
                    expt_register_usage_details = {
                        RegisterType.GPR  : (8, 6), # span of 8 registers (R0-R7), from which 6 are actually used
                        RegisterType.PRED : (1, 1),
                    }
                case 80 | 86 | 89:
                    expt_register_usage_details = {
                        RegisterType.GPR  : (8, 6),
                        RegisterType.PRED : (1, 1),
                        RegisterType.UGPR : (6, 2),
                    }
                case 90 | 100 | 120:
                    expt_register_usage_details = {
                        RegisterType.GPR  : (8, 7),
                        RegisterType.PRED : (1, 1),
                        RegisterType.UGPR : (7, 3),
                    }
                case _:
                    raise ValueError(f'unsupported {parameters.arch.compute_capability}')

            register_usage_details = disasm.functions[self.SYMBOL].registers
            assert register_usage_details == expt_register_usage_details

        def test_from_sass_annotated(self) -> None:
            """
            Read annotated SASS from file and check :py:meth:`reprospect.tools.binaries.NVDisasm.parse_sass_with_liveness_range_info`.
            """
            with self.SASS_ANNOTATED_FILE.open('r', encoding = 'utf-8') as fin:
                function = NVDisasm.parse_sass_with_liveness_range_info(
                    function_mangled = self.SYMBOL,
                    sass = iter(fin)
                )
            assert function.registers == {RegisterType.GPR : (8, 7), RegisterType.PRED : (1, 1), RegisterType.UGPR : (7, 3),}

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    class TestMany:
        """
        When there are many kernels.
        """
        CUDA_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'many.cu'
        CPP_FILE  : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'many.cpp'
        SYMBOLS   : typing.Final[tuple[str, ...]] = ('_Z6say_hiv', '_Z20vector_atomic_add_42PKfS0_Pfj')

        def test_from_executable(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
            """
            Compile :py:attr:`CPP_FILE` as an executable, extract cubin and run ``nvdisasm``.
            """
            output, _ = get_compilation_output(
                source = self.CPP_FILE,
                cwd = workdir,
                arch = parameters.arch,
                object_file = False,
                cmake_file_api = cmake_file_api,
            )

            _, cubin = CuObjDump.extract(
                file = output,
                arch = parameters.arch,
                cwd = workdir,
                cubin = get_cubin_name(
                    compiler_id = cmake_file_api.toolchains['CUDA']['compiler']['id'],
                    file = output,
                    arch = parameters.arch,
                    object_file = False,
                ),
            )

            disasm = NVDisasm(file = cubin, arch = parameters.arch)
            disasm.extract_register_usage_from_liveness_range_info(mangled = self.SYMBOLS)

            assert len(disasm.functions) == 2
            assert all(s in disasm.functions for s in self.SYMBOLS)

            match parameters.arch.compute_capability.as_int:
                case 70 | 75:
                    expt_register_usage_details = {
                        self.SYMBOLS[0] : {RegisterType.GPR : (22, 18), RegisterType.PRED : (2, 2),},
                        self.SYMBOLS[1] : {RegisterType.GPR : (12,  9), RegisterType.PRED : (1, 1),},
                    }
                case 80 | 86 | 89:
                    expt_register_usage_details = {
                        self.SYMBOLS[0] : {RegisterType.GPR : (22, 18), RegisterType.PRED : (2, 2),},
                        self.SYMBOLS[1] : {RegisterType.GPR : (12,  9), RegisterType.PRED : (1, 1), RegisterType.UGPR : (6, 2),},
                    }
                case 90 | 100:
                    expt_register_usage_details = {
                        self.SYMBOLS[0] : {RegisterType.GPR : (22, 18), RegisterType.PRED : (1, 1), RegisterType.UGPR : (8, 4),},
                        self.SYMBOLS[1] : {RegisterType.GPR : (12, 10), RegisterType.PRED : (1, 1), RegisterType.UGPR : (6, 2),},
                    }
                case 120:
                    expt_register_usage_details = {
                        self.SYMBOLS[0] : {RegisterType.GPR : (22, 18),                             RegisterType.UGPR : (8, 4),},
                        self.SYMBOLS[1] : {RegisterType.GPR : (12, 10), RegisterType.PRED : (1, 1), RegisterType.UGPR : (6, 2),},
                    }
                case _:
                    raise ValueError(f'unsupported {parameters.arch.compute_capability}')

            for symbol in self.SYMBOLS:
                register_usage_details = disasm.functions[symbol].registers
                assert register_usage_details == expt_register_usage_details[symbol]

    def test_string_representation(self) -> None:
        """
        Test :py:meth:`reprospect.tools.binaries.NVDisasm.__str__`.
        """
        def mock_init(
            self,
            file : pathlib.Path,
            arch : typing.Optional[NVIDIAArch] = None,
            demangler : typing.Type[CuppFilt | LlvmCppFilt] = CuppFilt
        ) -> None:
            self.file = file
            self.arch = arch
            self.demangler = demangler
            self.functions = {
                'my_kernel(float, const float *, float *, unsigned int)' : Function(
                    registers = TestFunction.REGISTERS
                ),
                'my_other_kernel(float, const float *, float *, unsigned int)' : Function(
                    registers = TestFunction.REGISTERS
                )
            }

        with unittest.mock.patch.object(NVDisasm, '__init__', mock_init):
            disasm = NVDisasm(file = pathlib.Path('code_object.1.sm_120.cubin'), arch = NVIDIAArch.from_str('BLACKWELL120'))

            assert str(disasm) == """\
NVDisasm of code_object.1.sm_120.cubin for architecture BLACKWELL120:
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ my_kernel(float, const float *, float *, unsigned int)                                                                             │
│ ┏━━━━━━┳━━━━━━━━━┳━━━━━━┓                                                                                                          │
│ ┃      ┃ Span    ┃ Used ┃                                                                                                          │
│ ┡━━━━━━╇━━━━━━━━━╇━━━━━━┩                                                                                                          │
│ │ GPR  │ R0-R7   │ 7    │                                                                                                          │
│ │ PRED │ P0      │ 1    │                                                                                                          │
│ │ UGPR │ UR0-UR6 │ 3    │                                                                                                          │
│ └──────┴─────────┴──────┘                                                                                                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ my_other_kernel(float, const float *, float *, unsigned int)                                                                       │
│ ┏━━━━━━┳━━━━━━━━━┳━━━━━━┓                                                                                                          │
│ ┃      ┃ Span    ┃ Used ┃                                                                                                          │
│ ┡━━━━━━╇━━━━━━━━━╇━━━━━━┩                                                                                                          │
│ │ GPR  │ R0-R7   │ 7    │                                                                                                          │
│ │ PRED │ P0      │ 1    │                                                                                                          │
│ │ UGPR │ UR0-UR6 │ 3    │                                                                                                          │
│ └──────┴─────────┴──────┘                                                                                                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
