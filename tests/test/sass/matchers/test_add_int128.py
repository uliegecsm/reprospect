import logging
import pathlib
import re
import subprocess
import typing

import pytest

from reprospect.test.sass.composite   import instructions_contain, instruction_is
from reprospect.test.sass.instruction import LoadGlobalMatcher, RegisterMatcher
from reprospect.test.sass.matchers    import add_int128
from reprospect.utils                 import cmake

from tests.parameters                 import Parameters, PARAMETERS
from tests.test.sass.test_instruction import get_decoder

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
class TestAddInt128:
    """
    Tests for :py:class:`reprospect.test.sass.matchers.add_int128.AddInt128`.
    """
    CODE_ADD_INT128: typing.Final[str] = """\
__global__ void add_int128(__int128_t* __restrict__ const dst, const __int128_t* __restrict__ const src)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] += src[index];
}
"""

    EXPECTED_PTX: typing.Final[re.Pattern[str]] = re.compile(r"""\
add\.cc\.s64 %rd\d+, %rd\d+, %rd\d+;
addc\.cc\.s64 %rd\d+, %rd\d+, %rd\d+;
st\.global\.v2\.(u|b)64 \[%rd\d+\], {%rd\d+, %rd\d+};
""")

    def test(self, request, parameters: Parameters, workdir: pathlib.Path, cmake_file_api: cmake.FileAPI) -> None:
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(self.CODE_ADD_INT128)

        decoder, output = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api, ptx = True)

        ptx = subprocess.check_output(('cuobjdump', '--dump-ptx', output)).decode()

        assert self.EXPECTED_PTX.search(ptx) is not None

        # Find the global load of the source.
        matcher_load_src = instructions_contain(matcher = LoadGlobalMatcher(arch = parameters.arch, size = 128, readonly = True))
        [matched_load_src] = matcher_load_src.assert_matches(instructions = decoder.instructions)

        logging.info(matched_load_src)

        # Find the global load of the destination.
        matcher_load_dst = instruction_is(matcher = LoadGlobalMatcher(arch = parameters.arch, size = 128, readonly = False)).times(1)
        [matched_load_dst] = matcher_load_dst.assert_matches(instructions = decoder.instructions[matcher_load_src.next_index:])

        logging.info(matched_load_dst)

        # Find the int128 addition pattern.
        matcher = add_int128.AddInt128()
        matched = matcher.assert_matches(instructions = decoder.instructions[matcher_load_src.next_index + 1:])

        logging.info(matched)

        assert matched[0].additional is not None
        assert 'start' in matched[0].additional

        reg_load  = RegisterMatcher(special = False).match(matched_load_src.operands[0])
        assert reg_load is not None
        reg_start = RegisterMatcher(special = False).match(matched[0].additional['start'][0])
        assert reg_start is not None

        assert reg_load.rtype == reg_start.rtype
        assert reg_load.index == reg_start.index
