import pathlib
import typing

import pytest

from reprospect.test.sass.composite import instructions_contain
from reprospect.test.sass.instruction import LoadGlobalMatcher
from reprospect.test.sass.matchers.add_int32 import AddInt32Matcher
from reprospect.utils import cmake

from tests.parameters import PARAMETERS, Parameters
from tests.test.sass.test_instruction import get_decoder


@pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
class TestAddInt32Matcher:
    """
    Tests for :py:class:`reprospect.test.sass.matchers.add_int32.AddInt32Matcher`.
    """
    CODE: typing.Final[str] = """\
__global__ void add({type}* __restrict__ const dst, const {type}* __restrict__ src)
{{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] += src[index];
}}
"""

    def test(self, request, parameters: Parameters, workdir: pathlib.Path, cmake_file_api: cmake.FileAPI) -> None:
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE.format(type='int32_t'))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api, ptx=True)

        matcher_ldg_src = instructions_contain(LoadGlobalMatcher(
            arch=parameters.arch, size=32, readonly=True,
        ))
        matcher_ldg_dst = instructions_contain(LoadGlobalMatcher(
            arch=parameters.arch, size=32, readonly=False,
        ))

        [matched_ldg_src] = matcher_ldg_src.assert_matches(instructions=decoder.instructions)
        [matched_ldg_dst] = matcher_ldg_dst.assert_matches(instructions=decoder.instructions[matcher_ldg_src.next_index:])

        offset = matcher_ldg_src.next_index + matcher_ldg_dst.next_index

        instructions_contain(AddInt32Matcher(
            arch=parameters.arch,
            src_a=matched_ldg_src.operands[0],
            src_b=matched_ldg_dst.operands[0],
            swap=False,
        )).assert_matches(instructions=decoder.instructions[offset:])

        instructions_contain(AddInt32Matcher(
            arch=parameters.arch,
            src_a=matched_ldg_dst.operands[0],
            src_b=matched_ldg_src.operands[0],
            swap=True,
        )).assert_matches(instructions=decoder.instructions[offset:])
