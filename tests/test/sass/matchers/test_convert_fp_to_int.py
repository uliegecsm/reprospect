import pathlib
import typing

import numpy
import pytest

from reprospect.test.sass.composite import instructions_contain
from reprospect.test.sass.matchers.convert_fp_to_int import ConvertFpToInt
from reprospect.utils.types import ConvertibleTypeInfo

from tests.parameters import PARAMETERS, Parameters
from tests.test.sass.test_instruction import get_decoder


@pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
class TestConvertFpToInt:
    """
    Tests for :py:class:`reprospect.test.sass.matchers.convert_fp_to_int.ConvertFpToInt`.
    """
    CODE: typing.Final[str] = """\
__global__ void test({int_type}* __restrict__ const dst, const {fp_type}* __restrict__ const src) {{
    dst[0] = src[0];
}}
"""

    @pytest.mark.parametrize(
        ('src_c_type', 'src_dtype', 'dst_c_type', 'dst_dtype'), [
            ('double', numpy.float64, 'int', numpy.int32),
            ('double', numpy.float64, 'long long', numpy.int64),
            ('double', numpy.float64, 'unsigned int', numpy.uint32),
            ('double', numpy.float64, 'unsigned long long', numpy.uint64),
            ('float', numpy.float32, 'int', numpy.int32),
            ('float', numpy.float32, 'long long', numpy.int64),
            ('float', numpy.float32, 'unsigned int', numpy.uint32),
            ('float', numpy.float32, 'unsigned long long', numpy.uint64),
    ], ids=str)
    def test(self, request, src_c_type: str, src_dtype: ConvertibleTypeInfo, dst_c_type: str, dst_dtype: ConvertibleTypeInfo, parameters: Parameters, workdir: pathlib.Path, cmake_file_api) -> None:
        FILE = workdir / f'{request.node.originalname}.{src_c_type}_to_{dst_c_type}.cu'
        FILE.write_text(self.CODE.format(fp_type=src_c_type, int_type=dst_c_type))

        decoder, _ = get_decoder(cwd=workdir, arch=parameters.arch, file=FILE, cmake_file_api=cmake_file_api)

        instructions_contain(ConvertFpToInt(src_dtype=src_dtype, dst_dtype=dst_dtype)).assert_matches(decoder.instructions)
