import logging
import os
import pathlib

import pytest
import typeguard

from cuda_helpers.tools import sass, binaries

from test_binaries import get_compilation_output, Parameters, PARAMETERS

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

@pytest.mark.parametrize("parameters", PARAMETERS, ids = str)
class TestSASSDeoder:
    """
    Test :py:class:`cuda_helpers.tools.sass.Decoder`.
    """
    FILE = pathlib.Path(__file__).parent / 'test_binaries' / 'saxpy.cu'
    
    @typeguard.typechecked
    def test(self, parameters : Parameters) -> None:
        """
        
        """
        output, _ = get_compilation_output(
            source = self.FILE,
            cwd = TMPDIR,
            arch = parameters.arch,
            object = True,
        )

        cuobjdump = binaries.CuObjDump(file = output, arch = parameters.arch, sass = True)

        source = pathlib.Path('test.sass')
        source.write_text(cuobjdump.sass)
        decoder = sass.Decoder(source = source)

        for instruction in decoder.instructions:
            logging.info(f'{instruction.instruction} {instruction.target_registers} {instruction.source_registers}')