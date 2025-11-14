import os
import pathlib
import typing

import pytest

from reprospect.tools.binaries     import CuObjDump, CuppFilt, ResourceType, NVDisasm
from reprospect.utils              import cmake

from tests.python.parameters          import Parameters, PARAMETERS
from tests.python.tools.test_binaries import get_compilation_output

@pytest.fixture(scope = 'session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

@pytest.fixture(scope = 'session')
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

class TestNVDisasm:
    """
    Tests related to :py:class:`reprospect.tools.binaries.NVDisasm`.
    .. note:
            In this test, we observe that ``cuobjdump`` indicates a register usage of 10 registers,
            whereas ``nvdisasm`` indicates that the span ``R0-R7``, hence a total of 8 registers,
            is used. The reason why ``cuobjdump`` indicates a use of 2 more registers than ``nvdisasm``
            is not clear, but it has been observed elsewhere too, see for instance:
            * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/UserGuide.md?plain=1#L262
    """
    @pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
    class TestSaxpy:
        """
        When the kernel performs a `saxpy`.
        """
        CUDA_FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent / 'assets' / 'saxpy.cu'
        SYMBOL    : typing.Final[str] = '_Z12saxpy_kernelfPKfPfj'
        SIGNATURE : typing.Final[str] = CuppFilt.demangle(SYMBOL)

        def test_nvdisasm_from_object(self, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
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

            # ``cuobjdump`` reports a register usage of 10 GPRs.
            assert cuobjdump.functions[self.SIGNATURE].ru[ResourceType.REGISTER] == 10

            # ``nvdisasm`` indicates that the span ``R0-R7`` of GPR registers is used.
            disasm = NVDisasm(file = cubin, arch = parameters.arch)
            disasm.extract_register_usage(mangled = [self.SYMBOL])

            assert len(disasm.functions) == 1
            assert self.SYMBOL in disasm.functions

            match parameters.arch.compute_capability.as_int:
                case 70 | 75:
                    expt_register_usage_details = {
                        'GPR' : (8, 6), # span of 8 GPR  registers (``R0-R7``), from which 6 GPR  registers are actually used
                        'PRED': (1, 1), # span of 1 PRED registers (``P0``),    from which 1 PRED register  is  actually used
                    }
                case 80 | 86 | 89:
                    expt_register_usage_details = {
                        'GPR' : (8, 6),
                        'PRED': (1, 1),
                        'UGPR': (6, 2),
                    }
                case 90 | 100 | 120:
                    expt_register_usage_details = {
                        'GPR' : (8, 7),
                        'PRED': (1, 1),
                        'UGPR': (7, 3),
                    }
                case _:
                    raise ValueError(f'unsupported {parameters.arch.compute_capability}')

            for reg_type in set(list(disasm.functions[self.SYMBOL].registers.keys()) + list(expt_register_usage_details.keys())):
                assert disasm.functions[self.SYMBOL].registers[reg_type] == expt_register_usage_details[reg_type]
