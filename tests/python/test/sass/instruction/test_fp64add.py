import typing

import pytest

from reprospect.test.sass.instruction import Fp64AddMatcher
from reprospect.utils                 import cmake

from tests.python.parameters                 import Parameters, PARAMETERS
from tests.python.test.sass.test_instruction import get_decoder

class TestFp64AddMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.Fp64AddMatcher`.
    """
    CODE_FP64_ADD : typing.Final[str] = """\
__global__ void fp64_add(double* __restrict__ const dst, const double* __restrict__ const src)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] += src[index];
}
"""

    def test(self) -> None:
        matched = Fp64AddMatcher().matches(inst = 'DADD R6, R4, R2')
        assert matched is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R6'
        assert matched.operands[-1] == 'R2'

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_from_compiled(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI):
        """
        Test with :py:attr:`CODE_FP64_ADD`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_FP64_ADD)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        matcher = Fp64AddMatcher()
        [dadd] = [(inst, matched) for inst in decoder.instructions if (matched := matcher.matches(inst))]
        inst, matched = dadd

        assert inst.instruction.startswith(matched.opcode)
        assert len(matched.operands) == 3
        assert all(operand in inst.instruction for operand in matched.operands)
        assert matched.additional is not None
        assert 'dst' in matched.additional and matched.operands[0] == matched.additional['dst'][0]
