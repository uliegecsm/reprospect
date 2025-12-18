import logging
import pathlib
import typing

import pytest

from reprospect.test.sass.controlflow.block import BasicBlockMatcher, BasicBlockWithParentMatcher
from reprospect.test.sass.composite import any_of, interleaved_instructions_are
from reprospect.test.sass.instruction import OpcodeModsMatcher, LoadGlobalMatcher
from reprospect.test.sass.instruction.half import Fp16MulMatcher, Fp16FusedMulAddMatcher
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass.controlflow import ControlFlow
from reprospect.tools.sass.decode import Decoder
from reprospect.utils import cmake

from tests.parameters import Parameters, PARAMETERS
from tests.test.sass.test_instruction import get_compilation_output

class TestBasicBlock:
    FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent.parent.parent / 'assets' / 'test_half.cu'

    SIGNATURE_INDIVIDUAL: typing.Final[str] = 'pow2_individual(__half *, const __half *, unsigned int)'
    SIGNATURE_PACKED: typing.Final[str] = 'pow2_packed(__half *, const __half *, unsigned int)'

    @pytest.fixture(scope='class')
    def cuobjdump(self, workdir: pathlib.Path, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> CuObjDump:
        output, _ = get_compilation_output(
            source=self.FILE,
            cwd=workdir,
            arch=parameters.arch,
            cmake_file_api=cmake_file_api,
        )

        cuobjdump = CuObjDump(file=output, arch=parameters.arch, sass=True)

        assert len(cuobjdump.functions) == 2

        return cuobjdump

@pytest.mark.parametrize('parameters', PARAMETERS, ids=str, scope='class')
class TestBasicBlockMatcher(TestBasicBlock):
    """
    Tests for :py:class:`reprospect.test.sass.controlflow.block.BasicBlockMatcher`.
    """
    def test_find_individual_and_packed_blocks(self, parameters: Parameters, cuobjdump: CuObjDump) -> None:
        """
        Find the "individual" block in :py:attr:`TestBasicBlock.SIGNATURE_INDIVIDUAL`, and both "individual"
        and "packed" blocks in :py:attr:`TestBasicBlock.SIGNATURE_PACKED`.

        Built on the observations from:

        * :py:meth:`tests.test.test_half.TestSASS.test_individual`
        * :py:meth:`tests.test.test_half.TestSASS.test_packed`
        """
        cfg_individual = ControlFlow.analyze(instructions=Decoder(code=cuobjdump.functions[self.SIGNATURE_INDIVIDUAL].code).instructions)
        cfg_packed     = ControlFlow.analyze(instructions=Decoder(code=cuobjdump.functions[self.SIGNATURE_PACKED].code).instructions)

        assert len(cfg_individual.blocks) == 3
        assert len(cfg_packed.blocks) == 5

        block_matcher_individual = BasicBlockMatcher(matcher=interleaved_instructions_are(
            LoadGlobalMatcher(arch=parameters.arch, size=16, readonly=True, extend='U'),
            Fp16MulMatcher(packed=False),
        ))

        block_matcher_individual.assert_matches(cfg_individual)
        matched_individual_from_packed, _ = block_matcher_individual.assert_matches(cfg_packed)

        block_matcher_packed = BasicBlockMatcher(matcher=interleaved_instructions_are(
            LoadGlobalMatcher(arch=parameters.arch, size=32, readonly=True),
            any_of(Fp16MulMatcher(packed=True), Fp16FusedMulAddMatcher(packed=True)),
        ))

        assert block_matcher_packed.match(cfg_individual) is None
        matched_packed_from_packed, _ = block_matcher_packed.assert_matches(cfg_packed)

        assert matched_packed_from_packed != matched_individual_from_packed

@pytest.mark.parametrize('parameters', PARAMETERS, ids=str, scope='class')
class TestBasicBlockWithParentMatcher(TestBasicBlock):
    """
    Tests for :py:class:`reprospect.test.sass.controlflow.block.BasicBlockWithParentMatcher`.
    """
    def test_find_individual_block(self, parameters: Parameters, cuobjdump: CuObjDump) -> None:
        """
        Find the "individual" block in :py:attr:`TestBasicBlock.SIGNATURE_INDIVIDUAL`, that is the child of a block with an ``ISETP`` instruction.
        """
        cfg_individual = ControlFlow.analyze(instructions=Decoder(code=cuobjdump.functions[self.SIGNATURE_INDIVIDUAL].code).instructions)

        assert len(cfg_individual.blocks) == 3

        block_isetp, [matched_isetp] = BasicBlockMatcher(OpcodeModsMatcher(
            opcode='ISETP', modifiers=('GE', 'U32', 'AND'), operands=True,
        )).assert_matches(cfg=cfg_individual)
        logging.info(matched_isetp)

        block_individual, [matched_ldg, matched_mul] = BasicBlockWithParentMatcher(parent=block_isetp, matcher=interleaved_instructions_are(
            LoadGlobalMatcher(arch=parameters.arch, size=16, readonly=True, extend='U'),
            Fp16MulMatcher(packed=False),
        )).assert_matches(cfg=cfg_individual)
        logging.info(matched_ldg)
        logging.info(matched_mul)

        assert block_isetp != block_individual
