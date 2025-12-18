import logging

import pytest

from reprospect.test.sass.instruction import Fp32AddMatcher
from reprospect.utils import cmake

from tests.parameters import PARAMETERS, Parameters
from tests.test.sass.test_instruction import (
    CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE,
    get_decoder,
)


class TestFp32AddMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.Fp32AddMatcher`.
    """
    def test(self) -> None:
        matched = Fp32AddMatcher().match(inst='FADD R6, R4, R2')
        assert matched is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R6'
        assert matched.operands[-1] == 'R2'

        matched = Fp32AddMatcher().match(inst='FADD R6, R4, c[0x0][0x178]')
        assert matched is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R6'
        assert matched.operands[-1] == 'c[0x0][0x178]'

        matched = Fp32AddMatcher().match(inst='FADD R25, R4, UR12')
        assert matched is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R25'
        assert matched.operands[-1] == 'UR12'

        matched = Fp32AddMatcher().match(inst='FADD.FTZ R9, -R7, 1.5707963705062866211')
        assert matched is not None
        assert matched.additional is not None
        assert matched.additional['dst'][0] == 'R9'
        assert matched.operands == ('R9', '-R7', '1.5707963705062866211')

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test_elementwise_add_restrict_wide(self, request, workdir, parameters: Parameters, cmake_file_api: cmake.FileAPI):
        """
        Test with :py:const:`tests.test.sass.test_instruction.CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE`.

        There will be 4 ``FADD`` instructions because of the :code:`float4`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE)

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        matcher = Fp32AddMatcher()
        fadd = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(fadd) == 4

        logging.info(matcher.pattern)

        for (inst, matched) in fadd:
            logging.info(inst.instruction)
            logging.info(matched)
            assert inst.instruction.startswith(matched.opcode)
            assert len(matched.operands) == 3
            assert all(operand in inst.instruction for operand in matched.operands)
            assert matched.additional is not None
            assert 'dst' in matched.additional and matched.additional['dst'][0] == matched.operands[0]
