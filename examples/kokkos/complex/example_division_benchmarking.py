import itertools
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
    ILogbScalbn = 'ILogbScalbn'
    Scaling = 'Scaling'

    @property
    def is_scaling(self) -> bool:
        return self == Method.Scaling

    @property
    def is_ilog(self) -> bool:
        return self == Method.ILogbScalbn

class Parameters(typing.TypedDict):
    method: Method
    branching_or_compliance: bool
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
        r'^NewtonFractal<Divisor(?P<divisor>Scaling|LogbScalbn)(?:<(?:(?P<branching_or_compliance>true|false))?(?:[, ]*(?P<ilogb>true|false))?>)?>/(?P<full>[A-Za-z]+)/width:(?P<width>\d+)/height:(?P<height>\d+)',
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
        matched = cls.PATTERN.search(name)

        assert matched is not None

        full: typing.Final[str] = matched.captures('full')[0]
        assert matched.captures('divisor')[0] in full

        branching_or_compliance: typing.Final[bool] = matched.captures('branching_or_compliance')[0] == 'true'
        height: typing.Final[int] = int(matched.captures('height')[0])
        width: typing.Final[int] = int(matched.captures('width')[0])

        match matched.captures('divisor')[0]:
            case 'LogbScalbn':
                method = Method.ILogbScalbn if matched.captures('ilogb')[0] == 'true' else Method.LogbScalbn
            case 'Scaling':
                method = Method.Scaling
            case _:
                raise ValueError(matched.captures('divisor'))

        reconstructed = ''.join((
            (f'NewtonFractal<Divisor{"Scaling" if method.is_scaling else "LogbScalbn"}'),
            ('<true' if branching_or_compliance else '<false'),
            ('>>' if method.is_scaling else f', {"true" if method.is_ilog else "false"}>>'),
            f'/{full}/width:{width}/height:{height}',
        ))

        assert name == reconstructed

        return {
            'branching_or_compliance': branching_or_compliance,
            'height': height,
            'method': method,
            'width': width,
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

        assert len(results) == len(Method) * 2 * 2

        # Legend.
        FONTSIZE = 22
        LINEWIDTH = 2

        MARKERS: typing.Final[dict[tuple[bool, bool], matplotlib.lines.Line2D]] = {
            (False, True):  matplotlib.lines.Line2D((0,), (0,), color='black', marker='s', linestyle='solid', lw=LINEWIDTH, markersize=10, markerfacecolor='grey', label='compliant'),
            (False, False): matplotlib.lines.Line2D((0,), (0,), color='black', marker='o', linestyle='dotted', lw=LINEWIDTH, markersize=10, markerfacecolor='grey', label='non-compliant'),
            (True, True): matplotlib.lines.Line2D((0,), (0,), color='black', marker='h', linestyle='solid', lw=LINEWIDTH, markersize=10, markerfacecolor='grey', label='branching'),
            (True, False): matplotlib.lines.Line2D((0,), (0,), color='black', marker='*', linestyle='dotted', lw=LINEWIDTH, markersize=10, markerfacecolor='grey', label='no branching'),
        }

        COLORS: typing.Final[dict[Method, matplotlib.lines.Line2D]] = {
            Method.Scaling:     matplotlib.lines.Line2D((0,), (0,), color='blue', lw=LINEWIDTH, linestyle='solid', label='scaling'),
            Method.LogbScalbn:  matplotlib.lines.Line2D((0,), (0,), color='red', lw=LINEWIDTH, linestyle='solid', label='logb-scalbn'),
            Method.ILogbScalbn: matplotlib.lines.Line2D((0,), (0,), color='green', lw=LINEWIDTH, linestyle='solid', label='ilogb-scalbn'),
        }

        # Make a nice plot.
        fig = matplotlib.pyplot.figure(figsize=(20, 10), layout='constrained')
        ax = fig.subplots(nrows=1, ncols=1)

        for (method, branching_or_compliance) in itertools.product(Method,(True, False)):
            filtered = results[
                (results['method'] == method) &
                (results['branching_or_compliance'] == branching_or_compliance)
            ].sort_values('size')
            marker = (method.is_scaling, branching_or_compliance)
            ax.plot(
                filtered['size'], filtered['real_time'],
                marker=MARKERS[marker].get_marker(),
                markersize=MARKERS[marker].get_markersize(),
                markerfacecolor=MARKERS[marker].get_markerfacecolor(),
                linestyle=MARKERS[marker].get_linestyle(),
                linewidth=MARKERS[marker].get_linewidth(),
                color=COLORS[method].get_color(),
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
