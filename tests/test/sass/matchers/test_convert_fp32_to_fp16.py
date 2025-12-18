import pathlib
import typing

import pytest

from reprospect.test.sass.composite import instructions_contain
from reprospect.test.sass.matchers.convert_fp32_to_fp16 import ConvertFp32ToFp16

from tests.parameters import Parameters, PARAMETERS
from tests.test.sass.test_instruction import get_decoder

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
class TestConvertFp32ToFp16:
    """
    Tests for :py:class:`reprospect.test.sass.matchers.convert_fp32_to_fp16.ConvertFp32ToFp16`.
    """
    CODE: typing.Final[str] = """#include "cuda_fp16.h"
__global__ void test(__half* __restrict__ const dst, const float* __restrict__ const src) {
    dst[0] = __float2half(src[0]);
}
"""

    def test(self, request, parameters: Parameters, workdir: pathlib.Path, cmake_file_api) -> None:
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        instructions_contain(ConvertFp32ToFp16(arch=parameters.arch)).assert_matches(decoder.instructions)
