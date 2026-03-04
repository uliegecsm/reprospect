import logging
import os
import pathlib
import subprocess
import sys
import typing

import matplotlib.pyplot
import numpy
import pytest

from reprospect.test.case import CMakeAwareTestCase
from reprospect.utils import detect

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestDivision(CMakeAwareTestCase):
    EXTENT: typing.Final[tuple[int, ...]] = (-1, 1, -1, 1)

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_complex_division_plot'

    @pytest.fixture(scope='class')
    def data(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Run the executable and read the output arrays.
        """
        env = os.environ.copy()
        env.update({'OUTPUT_DIR': str(self.cwd)})

        logging.info(f'Executing {self.executable}.')
        subprocess.check_call(self.executable, env=env)

        colors_file = self.cwd / 'colors.bin'
        logging.info(f'Loading colors from {colors_file}.')
        with colors_file.open('rb') as fin:
            dim_0 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
            dim_1 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
            colors = numpy.fromfile(fin, dtype=numpy.uint32).reshape((dim_1, dim_0))

        iterations_file = self.cwd / 'iterations.bin'
        logging.info(f'Loading iterations from {iterations_file}.')
        with iterations_file.open('rb') as fin:
            dim_0 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
            dim_1 = numpy.fromfile(fin, dtype=numpy.uint64, count=1)[0]
            iterations = numpy.fromfile(fin, dtype=numpy.uint32).reshape((dim_1, dim_0))

        assert colors.shape == iterations.shape

        return (colors, iterations)

    def test(self, data: tuple) -> None:
        """
        Plot the colors and iterations with some artist touch.
        """
        fig, axes = matplotlib.pyplot.subplots(
            nrows=1, ncols=2,
            figsize=(20, 10),
            constrained_layout=True,
        )

        fig.patch.set_facecolor('black')

        # Colors.
        axes[0].imshow(
            data[0],
            origin='lower',
            extent=self.EXTENT,
        )

        axes[0].set_title('Based on root index', color='white')

        # Iterations.
        bands = 24
        periodic = numpy.mod(data[1].astype(float), bands)
        axes[1].imshow(
            periodic,
            origin='lower',
            extent=self.EXTENT,
            cmap='turbo',
            interpolation='bilinear',
        )

        axes[1].set_title('Based on number of iterations', color='white')

        ARTIFACT_DIR = pathlib.Path(os.environ['ARTIFACT_DIR'])
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        output = ARTIFACT_DIR / 'fractal.svg'
        logging.info(f'Writing to {output}.')
        matplotlib.pyplot.savefig(output, bbox_inches='tight')
