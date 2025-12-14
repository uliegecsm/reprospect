import pathlib
import typing

import pytest

from reprospect.test.sass.controlflow.block import BlockMatcher
from reprospect.test.sass.composite import any_of, interleaved_instructions_are
from reprospect.test.sass.instruction import LoadGlobalMatcher
from reprospect.test.sass.instruction.half import Fp16MulMatcher, Fp16FusedMulAddMatcher
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.sass.controlflow import ControlFlow
from reprospect.tools.sass.decode import Decoder
from reprospect.utils import cmake

from tests.parameters import Parameters, PARAMETERS
from tests.test.sass.test_instruction import get_compilation_output

@pytest.mark.parametrize('parameters', PARAMETERS, ids=str, scope='class')
class TestBlockMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.controlflow.block.BlockMatcher`.
    """
    FILE : typing.Final[pathlib.Path] = pathlib.Path(__file__).parent.parent.parent.parent / 'assets' / 'test_half.cu'

    SIGNATURE_INDIVIDUAL : typing.Final[str] = 'pow2_individual(__half *, const __half *, unsigned int)'
    SIGNATURE_PACKED : typing.Final[str] = 'pow2_packed(__half *, const __half *, unsigned int)'

    @pytest.fixture(scope='class')
    def cuobjdump(self, workdir : pathlib.Path, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> CuObjDump:
        output, _ = get_compilation_output(
            source=self.FILE,
            cwd=workdir,
            arch=parameters.arch,
            cmake_file_api=cmake_file_api,
        )

        cuobjdump = CuObjDump(file=output, arch=parameters.arch, sass=True)

        assert len(cuobjdump.functions) == 2

        return cuobjdump

    def test_find_individual_and_packed_blocks(self, parameters: Parameters, cuobjdump: CuObjDump) -> None:
        """
        Find the "individual" block in :py:attr:`SIGNATURE_INDIVIDUAL`, and both "individual"
        and "packed" blocks in :py:attr:`SIGNATURE_PACKED`.

        Built on the observations from:

        * :py:meth:`tests.test.test_half.TestSASS.test_individual`
        * :py:meth:`tests.test.test_half.TestSASS.test_packed`
        """
        cfg_individual = ControlFlow.analyze(instructions=Decoder(code=cuobjdump.functions[self.SIGNATURE_INDIVIDUAL].code).instructions)
        cfg_packed     = ControlFlow.analyze(instructions=Decoder(code=cuobjdump.functions[self.SIGNATURE_PACKED].code).instructions)

        assert len(cfg_individual.blocks) == 3
        assert len(cfg_packed.blocks) == 5

        block_matcher_individual = BlockMatcher(matcher=interleaved_instructions_are(
            LoadGlobalMatcher(arch=parameters.arch, size=16, readonly=True, extend='U'),
            Fp16MulMatcher(packed=False),
        ))

        block_matcher_individual.assert_matches(cfg_individual)
        matched_individual_from_packed, _ = block_matcher_individual.assert_matches(cfg_packed)

        block_matcher_packed = BlockMatcher(matcher=interleaved_instructions_are(
            LoadGlobalMatcher(arch=parameters.arch, size=32, readonly=True),
            any_of(Fp16MulMatcher(packed=True), Fp16FusedMulAddMatcher(packed=True)),
        ))

        assert block_matcher_packed.match(cfg_individual) is None
        matched_packed_from_packed, _ = block_matcher_packed.assert_matches(cfg_packed)

        assert matched_packed_from_packed != matched_individual_from_packed
