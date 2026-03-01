"""
Many bla bla
"""

import itertools
import json
import logging
import pathlib
import re
import subprocess
import sys
import typing

import matplotlib.artist
import matplotlib.legend
import matplotlib.legend_handler
import matplotlib.lines
import matplotlib.pyplot
import matplotlib.text
import matplotlib.transforms
import numpy
import pandas
import pytest

from reprospect.test import CMakeAwareTestCase
from reprospect.utils import detect

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Method(StrEnum):
    IEC559  = 'iec559'
    SCALING = 'scaling'
    SCALING_BRANCH = 'scaling_branch'

class Parameters(typing.TypedDict):
    method: Method
    count: int

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestDivision(CMakeAwareTestCase):
    """
    Run the companion executable and make a nice visualization.
    """
    TIME_UNIT: typing.Final = 'ns'
    """
    Time unit of the benchmark.
    """

    PATTERN: typing.Final[re.Pattern[str]] = re.compile(
        r'^Division/([A-Za-z0-9_]+)/size:([0-9]+)',
    )

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_complex_division_benchmarking'

    @classmethod
    def params(cls, *, name: str) -> Parameters:
        """
        Parse the name of a case and return parameters.
        """
        match = cls.PATTERN.search(name)

        assert match is not None
        assert len(match.groups()) == 2

        return {
            'method': Method(match.group(1)),
            'size': int(match.group(2)),
        }

    @pytest.fixture(scope='class')
    def raw(self) -> dict[str, dict]:
        """
        Run the benchmark and return the raw `JSON`-based results.

        .. warning::

            Be sure to remove `--benchmark_min_time` for better converged results.
        """
        file: pathlib.Path = self.cwd / 'results.json'

        cmd: tuple[str | pathlib.Path, ...] = (
            self.executable,
            # '--benchmark_min_time=2x',
            '--benchmark_enable_random_interleaving=true',
            f'--benchmark_out={file}',
            '--benchmark_out_format=json',
        )

        logging.info(f'Running benchmark with {cmd}.')

        subprocess.check_call(cmd, cwd=self.cwd)

        with file.open(mode='r') as fp:
            return json.load(fp=fp)

    @pytest.fixture(scope='class')
    def results(self, raw: dict) -> pandas.DataFrame:
        """
        Processed results.
        """
        def process(bench_case) -> dict[str, Method | int]:
            logging.debug(f'Processing benchmark case {bench_case["name"]}.')
            assert bench_case['time_unit'] == self.TIME_UNIT
            return {
                **self.params(name=bench_case['name']),
                'real_time': bench_case['real_time'],
            }

        return pandas.DataFrame(
            process(bench_case)
            for bench_case in raw['benchmarks']
        )

    def test_visualize(self, results: pandas.DataFrame) -> None:
        """
        Create a visualization of the results.
        """
        # Retrieve unique, sorted sizes.
        sizes = sorted(set(results['size'].values))
        assert len(sizes) == 5

        logging.info(f'Sizes are {sizes}.')

        # Legend.
        FONTSIZE = 22

        LINESTYLES: typing.Final[dict[bool, matplotlib.lines.Line2D]] = {
            Method.IEC559: matplotlib.lines.Line2D((0,), (0,), color='black', linestyle='solid',  lw=4, label='async'),
            Method.SCALING: matplotlib.lines.Line2D((0,), (0,), color='red', linestyle='dotted', lw=4, label='sync'),
            Method.SCALING_BRANCH: matplotlib.lines.Line2D((0,), (0,), color='blue', linestyle='dashed', lw=4, label='sync'),
        }

        # Make a nice plot.
        fig = matplotlib.pyplot.figure(figsize=(20, 10), layout='constrained')
        ax  = fig.subplots(nrows=1, ncols=1)

        for method in Method:
            filtered = results[
                (results['method'] == method)
            ].sort_values('size')
            ax.plot(
                filtered['size'], filtered['real_time'],
                linestyle=LINESTYLES[method].get_linestyle(),
                linewidth=LINESTYLES[method].get_linewidth(),
                color=LINESTYLES[method].get_color(),
                label=f'{method}',
            )

        ax.set_ylabel(f'time [{self.TIME_UNIT}]',   fontsize=FONTSIZE)
        ax.set_xlabel(f'number of elements', fontsize=FONTSIZE)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE)

        ax.grid(which='both')
        ax.legend(fontsize=FONTSIZE)

        fname = self.cwd / 'results.svg'
        logging.info(f'Saving results in {fname}.')
        fig.savefig(fname=fname, bbox_inches='tight', transparent=False)

class HandleSubtitle(matplotlib.legend_handler.HandlerBase):
    @override
    def create_artists(self,
        legend: matplotlib.legend.Legend,
        orig_handle: matplotlib.artist.Artist,
        xdescent: float, ydescent: float,
        width: float, height: float,
        fontsize: float, trans: matplotlib.transforms.Transform,
    ) -> list[matplotlib.artist.Artist]:
        if not isinstance(orig_handle, Subtitle):
            raise TypeError('Wrong usage.')

        text = matplotlib.text.Text(
            x=xdescent, y=ydescent,
            text=orig_handle.text,
            fontsize=fontsize, transform=trans,
        )

        return [text]

class Subtitle:
    def __init__(self, text: str):
        self.text: typing.Final[str] = text

    def get_label(self) -> str:
        return ''
