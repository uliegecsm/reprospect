import json
import logging
import pathlib
import subprocess
import sys
import typing

import matplotlib.lines
import matplotlib.pyplot
import matplotlib.text
import pandas
import pytest
import regex

from reprospect.test import CMakeAwareTestCase
from reprospect.utils import detect

from examples.kokkos.pyplot import HandlerText

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Method(StrEnum):
    LogbScalbn = 'LogbScalbn'
    LogbScalbnBranch = 'LogbScalbnBranch'
    Scaling = 'Scaling'
    ScalingBranch = 'ScalingBranch'

    @property
    def has_branching(self) -> bool:
        return 'Branch' in self.name

    @property
    def is_scaling(self) -> bool:
        return 'Scaling' in self.name

class Parameters(typing.TypedDict):
    method: Method
    width: int
    height: int

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason='needs a GPU')
class TestDivision(CMakeAwareTestCase):
    """
    Run the companion executable and make a nice visualization.
    """
    TIME_UNIT: typing.Final = 'us'
    """
    Time unit of the benchmark.
    """

    PATTERN: typing.Final[regex.Pattern[str]] = regex.compile(
        r'^NewtonFractal<Divisor(Scaling|LogbScalbn)<(true|false)>>/(?P<method>[A-Za-z]+)/width:(?P<width>[0-9]+)/height:(?P<height>[0-9]+)',
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
        assert len(match.groups()) == 5
        assert match.groups()[2].startswith(match.groups()[0])

        method = Method(match.captures('method')[0])
        branching = match.group(2) == 'true'
        scaling = match.group(1) == 'Scaling'

        assert branching == method.has_branching
        assert scaling == method.is_scaling

        return {
            'method': method,
            'width': int(match.captures('width')[0]),
            'height': int(match.captures('height')[0]),
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
            '--benchmark_min_time=1x',
            '--benchmark_enable_random_interleaving=true',
            f'--benchmark_out={file}',
            '--benchmark_out_format=json',
            f'--benchmark_time_unit={self.TIME_UNIT}',
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
        # Add a column with the size, computed from width and height.
        results['size'] = results['width'] * results['height']

        assert len(results) == len(Method) * 2

        # Legend.
        FONTSIZE = 22
        LINEWIDTH = 2

        MARKERS: typing.Final[dict[bool, matplotlib.lines.Line2D]] = {
            True:  matplotlib.lines.Line2D((0,), (0,), color='black', marker='s', linestyle='solid', lw=LINEWIDTH, markersize=10, markerfacecolor='grey', label='yes'),
            False: matplotlib.lines.Line2D((0,), (0,), color='black', marker='o', linestyle='dotted', lw=LINEWIDTH, markersize=10, markerfacecolor='grey', label='no'),
        }

        COLORS: typing.Final[dict[bool, matplotlib.lines.Line2D]] = {
            True:  matplotlib.lines.Line2D((0,), (0,), color='blue', lw=LINEWIDTH, linestyle='solid', label='scaling'),
            False: matplotlib.lines.Line2D((0,), (0,), color='red', lw=LINEWIDTH, linestyle='solid', label='logb-scalbn'),
        }

        # Make a nice plot.
        fig = matplotlib.pyplot.figure(figsize=(20, 10), layout='constrained')
        ax = fig.subplots(nrows=1, ncols=1)

        for method in Method:
            filtered = results[(results['method'] == method)].sort_values('size')
            ax.plot(
                filtered['size'], filtered['real_time'],
                marker=MARKERS[method.has_branching].get_marker(),
                markersize=MARKERS[method.has_branching].get_markersize(),
                markerfacecolor=MARKERS[method.has_branching].get_markerfacecolor(),
                linestyle=MARKERS[method.has_branching].get_linestyle(),
                linewidth=MARKERS[method.has_branching].get_linewidth(),
                color=COLORS[method.is_scaling].get_color(),
            )

        ax.set_ylabel(f'time [{self.TIME_UNIT}]', fontsize=FONTSIZE)
        ax.set_xlabel('number of elements', fontsize=FONTSIZE)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE)
        ax.grid(which='both')

        _ = fig.legend(handles=(
                matplotlib.text.Text(text='Method'),
                *COLORS.values(),
                matplotlib.text.Text(text=''),
                matplotlib.text.Text(text='Branching'),
                *MARKERS.values(),
            ),
            loc='outside center left',
            frameon=False,
            handler_map={matplotlib.text.Text: HandlerText()},
            fontsize=FONTSIZE,
        )

        fname = self.cwd / 'results.svg'
        logging.info(f'Saving results in {fname}.')
        fig.savefig(fname=fname, bbox_inches='tight', transparent=False)
