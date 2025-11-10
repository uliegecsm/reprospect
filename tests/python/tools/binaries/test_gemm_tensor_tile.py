import os
import pathlib
import subprocess

import numpy
import numpy.typing
import pytest

from reprospect.utils import cmake, detect

from tests.python.parameters import PARAMETERS

from tests.python.tools.binaries.assets.gemm_tensor_tile import GemmTensorTile

@pytest.fixture(scope = 'session')
def workdir() -> pathlib.Path:
    return pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])

@pytest.fixture(scope = 'session')
def cmake_file_api() -> cmake.FileAPI:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR']),
    )

@pytest.mark.parametrize('parameters', PARAMETERS, ids = str, scope = 'class')
class TestGemmTensorTile:
    """
    Test correctness of :py:class:`tests.python.tools.binaries.assets.gemm_tensor_tile.GemmTensorTile`.
    """
    @pytest.fixture(scope = 'class')
    def skipif(self, parameters) -> None:
        """
        The test needs a GPU of the correct architecture.
        """
        if parameters.arch not in detect.GPUDetector.detect()['architecture'].values:
            pytest.skip(f'This test requires {parameters.arch} but got {detect.GPUDetector.detect()["architecture"].values}.')

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_correctness(self, skipif, workdir : pathlib.Path, parameters, cmake_file_api) -> None:
        """
        Ensure correctness of the algorithm.
        """
        # Dimensions must be multiples of 16.
        FACTOR = 4
        M, N, K = 16 * FACTOR, 16 * FACTOR, 16 * FACTOR

        # Generate input matrices.
        mat_A : numpy.typing.NDArray[numpy.float16] = numpy.random.rand(M, K).astype(numpy.float16)
        mat_B : numpy.typing.NDArray[numpy.float16] = numpy.random.rand(K, N).astype(numpy.float16)

        # Save to binary.
        mat_A.tofile(workdir / 'mat_A.bin')
        mat_B.tofile(workdir / 'mat_B.bin')

        # Execute.
        subprocess.check_call([
            GemmTensorTile.executable(arch = parameters.arch, cwd = workdir, cmake_file_api = cmake_file_api),
            str(M), str(N), str(K),
            workdir / 'mat_A.bin',
            workdir / 'mat_B.bin',
            workdir / 'mat_C.bin',
        ])

        # Read result and check.
        C_computed = numpy.fromfile(workdir / 'mat_C.bin', dtype = numpy.float32).reshape(M, N)

        C_reference = numpy.matmul(mat_A, mat_B)

        assert numpy.allclose(
            a = C_reference,
            b = C_computed,
            rtol = 1.e-3,
            atol = 1.e-8,
        )
