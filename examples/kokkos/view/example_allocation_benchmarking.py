"""
Comparing :code:`Kokkos::View` allocation against native CUDA implementation
============================================================================

:code:`cudaMallocAsync` calls are immediately followed by a stream or device synchronization, as seen in
https://github.com/kokkos/kokkos/blob/146241cf3a68454527994a46ac473861c2b5d4f1/core/src/Cuda/Kokkos_CudaSpace.cpp#L209-L220.

Moreover, :code:`Kokkos` opts not to use :code:`cudaMallocAsync` when allocation sizes fall below a threshold defined at
https://github.com/kokkos/kokkos/blob/c1a715cab26da9407867c6a8c04b2a1d6b2fc7ba/core/src/impl/Kokkos_SharedAlloc.hpp#L23.

Additionally, at least for its CUDA backend, :code:`Kokkos` copies a *shared allocation header* (mainly used for debug), as seen *e.g.* in
https://github.com/kokkos/kokkos/blob/bba2d1f60741b6a2023b36313016c0a0dd125f42/core/src/impl/Kokkos_SharedAlloc.hpp#L325-L327.
This copy operation invariably uses :code:`cudaMemcpyAsync`, as demonstrated in :py:class:`examples.kokkos.view.example_allocation_tracing.TestNSYS`.
Consequently, :code:`Kokkos` will always:

1. allocate buffers that are 128 bytes larger than requested
2. entail additional API calls

This copy of the shared allocation header has triggered the discussion at
https://github.com/kokkos/kokkos/issues/8441.

The benchmark results clearly show that allocating with :code:`Kokkos` consistently incurs additional overhead,
compared to a native CUDA implementation.

References:

* https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-memory-allocator
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

from reprospect.test  import CMakeAwareTestCase
from reprospect.utils import detect

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Framework(StrEnum):
    CUDA = 'CUDA'
    KOKKOS = 'Kokkos'

class Parameters(typing.TypedDict):
    use_async: bool
    framework: Framework
    count: int
    size: int

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestAllocation(CMakeAwareTestCase):
    """
    Run the companion executable and make a nice visualization.
    """
    TIME_UNIT:typing.Final = 'ns'
    """
    Time unit of the benchmark.
    """

    THRESHOLD: typing.Final[int] = 40000
    """
    Threshold for using stream ordered allocation, see https://github.com/kokkos/kokkos/blob/146241cf3a68454527994a46ac473861c2b5d4f1/core/src/Cuda/Kokkos_CudaSpace.cpp#L147.
    """

    PATTERN: typing.Final[re.Pattern[str]] = re.compile(
        r'^With(CUDA|Kokkos)<(true|false)>/((?:cuda|kokkos)(?:_async)?)/count:([0-9]+)/size:([0-9]+)',
    )

    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_view_allocation_benchmarking'

    @classmethod
    def params(cls, *, name: str) -> Parameters:
        """
        Parse the name of a case and return parameters.
        """
        match = cls.PATTERN.search(name)

        assert match is not None
        assert len(match.groups()) == 5
        assert match.group(1) in Framework
        assert match.group(2) in ['true', 'false']

        framework = Framework(match.group(1))
        use_async = match.group(2) == 'true'

        expt_name = f'{framework.lower()}' + ('_async' if use_async else '')

        assert match.group(3) == expt_name

        return {
            'framework': framework,
            'use_async': use_async,
            'count': int(match.group(4)),
            'size': int(match.group(5)),
        }

    @pytest.fixture(scope = 'class')
    def raw(self) -> dict[str, dict]:
        """
        Run the benchmark and return the raw `JSON`-based results.

        .. warning::

            Be sure to remove `--benchmark_min_time` for better converged results.
        """
        file: pathlib.Path = self.cwd / 'results.json'

        cmd: tuple[str | pathlib.Path, ...] = (
            self.executable,
            '--benchmark_min_time=2x',
            '--benchmark_enable_random_interleaving=true',
            f'--benchmark_out={file}',
            '--benchmark_out_format=json',
        )

        logging.info(f'Running benchmark with {cmd}.')

        subprocess.check_call(cmd, cwd = self.cwd)

        with file.open(mode = 'r') as fp:
            return json.load(fp = fp)

    @pytest.fixture(scope = 'class')
    def results(self, raw: dict) -> pandas.DataFrame:
        """
        Processed results.
        """
        def process(bench_case) -> dict[str, Framework | bool | int | float]:
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

    def test_memory_pool_attributes(self, raw) -> None:
        """
        Retrieve the memory pool attributes and check consistency and behavior.
        """
        COLUMNS: typing.Final[tuple[str, ...]] = (
            'ReservedMemCurrent',
            'ReservedMemHigh',
            'UsedMemCurrent',
            'UsedMemHigh',
            'ReleaseThreshold',
        )
        attributes = pandas.DataFrame(
            data = (
                tuple(int(bench_case['cudaMemPoolAttr' + attr]) for attr in COLUMNS)
                for bench_case in raw['benchmarks']
            ),
            columns = COLUMNS,
        )

        # After the test, the current reserved/used memory is the same for everyone.
        assert attributes['ReservedMemCurrent'].nunique(dropna = False) == 1
        assert attributes[    'UsedMemCurrent'].nunique(dropna = False) == 1

        logging.info(f"Default memory pool reserved size is always {attributes.loc[0, 'ReservedMemCurrent']}.")
        logging.info(f"Default memory pool used     size is always {attributes.loc[0,     'UsedMemCurrent']}.")

        # The release threshold is 0 for everyone.
        assert attributes['ReleaseThreshold'].eq(0).all()

        logging.info('The release threshold is always 0.')

        logging.info(f"Reserved high: {sorted(set(attributes['ReservedMemHigh']))}")
        logging.info(f"Used     high: {sorted(set(attributes[    'UsedMemHigh']))}")

        # The memory pool grows by taking steps of multiples of 32MiB.
        STEP_SIZE = 1024**2 * 32
        steps = numpy.diff(numpy.sort(attributes['ReservedMemHigh'].unique()))
        assert (steps % STEP_SIZE == 0).all()

    def test_visualize(self, results: pandas.DataFrame) -> None:
        """
        Create a visualization of the results.
        """
        # Retrieve unique, sorted counts.
        counts = sorted(set(results['count'].values))
        assert len(counts) == 4

        logging.info(f'Counts are {counts}.')

        # Factor for size (results are initially in bytes).
        SIZE_FACTOR: typing.Final[int] = 1000
        SIZE_UNIT: typing.Final[str] = 'kB'

        # Legend.
        FONTSIZE = 22

        LINESTYLES: typing.Final[dict[bool, matplotlib.lines.Line2D]] = {
            True:  matplotlib.lines.Line2D((0,), (0,), color = 'black', linestyle = 'solid',  lw = 4, label = 'async'),
            False: matplotlib.lines.Line2D((0,), (0,), color = 'black', linestyle = 'dotted', lw = 4, label = 'sync'),
        }

        MARKERS: typing.Final[dict[Framework, matplotlib.lines.Line2D]] = {
            Framework.CUDA:   matplotlib.lines.Line2D((0,), (0,), color = 'black', marker = 's', linestyle = '', markersize = 10, markerfacecolor = 'grey', label = 'CUDA'),
            Framework.KOKKOS: matplotlib.lines.Line2D((0,), (0,), color = 'black', marker = 'o', linestyle = '', markersize = 10, markerfacecolor = 'grey', label = 'Kokkos'),
        }

        COLORS: typing.Final[dict[int, matplotlib.lines.Line2D]] = {
            1:  matplotlib.lines.Line2D((0,), (0,), color = 'black', lw = 2, linestyle = 'solid', label = '1'),
            4:  matplotlib.lines.Line2D((0,), (0,), color = 'red',   lw = 2, linestyle = 'solid', label = '4'),
            8:  matplotlib.lines.Line2D((0,), (0,), color = 'green', lw = 2, linestyle = 'solid', label = '8'),
            12: matplotlib.lines.Line2D((0,), (0,), color = 'blue',  lw = 2, linestyle = 'solid', label = '12'),
        }

        # Make a nice plot.
        fig = matplotlib.pyplot.figure(figsize = (40, 20), layout = 'constrained')
        ax  = fig.subplots(nrows = 1, ncols = 1)

        threshold = ax.axvline(self.THRESHOLD / SIZE_FACTOR, color = 'black', linestyle = 'dashed', label = 'threshold', lw = 3)

        for framework, use_async in itertools.product(tuple(Framework), (True, False)):
            for count in counts:
                filtered = results[
                    (results['count'] == count) &
                    (results['framework'] == framework) &
                    (results['use_async'] == use_async)
                ].sort_values('size')
                ax.plot(
                    filtered['size'] / SIZE_FACTOR, filtered['real_time'],
                    marker = MARKERS[framework].get_marker(),
                    markersize = MARKERS[framework].get_markersize(),
                    markerfacecolor = MARKERS[framework].get_markerfacecolor(),
                    linestyle = LINESTYLES[use_async].get_linestyle(),
                    linewidth = LINESTYLES[use_async].get_linewidth(),
                    color = COLORS[count].get_color(),
                )

        ax.set_ylabel(f'time [{self.TIME_UNIT}]',   fontsize = FONTSIZE)
        ax.set_xlabel(f'buffer size [{SIZE_UNIT}]', fontsize = FONTSIZE)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis = 'both', which = 'major', labelsize = FONTSIZE)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = FONTSIZE)

        _ = fig.legend(handles = (
                Subtitle(text = 'Framework'),
                *MARKERS.values(),
                Subtitle(text = ''),
                *LINESTYLES.values(),
                Subtitle(text = ''),
                Subtitle(text = 'Count'),
                *COLORS.values(),
                Subtitle(text = ''),
                threshold,
            ),
            loc = 'outside center left',
            frameon = False,
            handler_map = {Subtitle: HandleSubtitle()},
            fontsize = FONTSIZE,
        )

        ax.grid(which = 'both')

        fname = self.cwd / 'results.svg'
        logging.info(f'Saving results in {fname}.')
        fig.savefig(fname = fname, bbox_inches = 'tight', transparent = False)

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
            x = xdescent, y = ydescent,
            text = orig_handle.text,
            fontsize = fontsize, transform = trans,
        )

        return [text]

class Subtitle:
    def __init__(self, text: str):
        self.text: typing.Final[str] = text

    def get_label(self) -> str:
        return ''
