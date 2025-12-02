import logging
import os
import pathlib
import re
import subprocess
import tempfile
import typing
import unittest.mock

import pytest

from reprospect.test.cmake import get_demangler_for_compiler

from reprospect.tools import ncu
from reprospect.utils import detect

@pytest.fixture(scope = 'session')
def cmake_cuda_compiler(cmake_file_api) -> dict:
    return cmake_file_api.toolchains['CUDA']['compiler']

class TestProfilingResults:
    """
    Tests for :py:class:`reprospect.tools.ncu.ProfilingResults`.
    """
    @pytest.fixture(scope = 'class')
    def results(self) -> ncu.ProfilingResults:
        results = ncu.ProfilingResults()
        results.assign_metrics(
            accessors = ('nvtx_range_name', 'nvtx_push_region_A', 'nvtx_push_region_kernel', 'kernel'),
            data = {
                'smsp__inst_executed.sum' : 100.,
                'sass__inst_executed_per_opcode': ncu.MetricCorrelationData(correlated = {'LDCU': 16., 'LDC': 16.,}),
                'L1/TEX cache global load sectors.sum' : 0.
            }
        )
        results.assign_metrics(
            accessors = ('nvtx_range_name', 'nvtx_push_region_B', 'nvtx_push_region_other_kernel', 'other_kernel'),
            data = {
                'smsp__inst_executed.sum' : 96.,
                'sass__inst_executed_per_opcode': ncu.MetricCorrelationData(correlated = {'LDCU': 16., 'LDC': 16.,}),
                'L1/TEX cache global load sectors.sum' : 0.
            }
        )
        return results

    def test_query(self, results : ncu.ProfilingResults) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.ProfilingResults.query`.
        """
        with pytest.raises(KeyError, match = "'nvtx_range_not_in_results'"):
            results.query(('nvtx_range_not_in_results',))

        results_A = results.query(('nvtx_range_name', 'nvtx_push_region_A'))
        assert isinstance(results_A, ncu.ProfilingResults)

        metrics_A_kernel = results.query(('nvtx_range_name', 'nvtx_push_region_A', 'nvtx_push_region_kernel', 'kernel'))
        assert isinstance(metrics_A_kernel, ncu.ProfilingMetrics)
        assert metrics_A_kernel['smsp__inst_executed.sum'] == 100.

    def test_query_metrics(self, results : ncu.ProfilingResults) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.ProfilingResults.query_metrics`.
        """
        metrics = results.query_metrics(('nvtx_range_name', 'nvtx_push_region_A', 'nvtx_push_region_kernel', 'kernel'))
        assert metrics['smsp__inst_executed.sum'] == 100.

    def test_query_single_next(self, results : ncu.ProfilingResults) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.ProfilingResults.query_single_next`.
        """
        key, metrics = results.query_single_next(('nvtx_range_name', 'nvtx_push_region_A', 'nvtx_push_region_kernel',))
        assert key == 'kernel'
        assert metrics == results.query(('nvtx_range_name', 'nvtx_push_region_A', 'nvtx_push_region_kernel', 'kernel',))

    def test_query_single_next_metrics(self, results : ncu.ProfilingResults) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.ProfilingResults.query_single_next_metrics`.
        """
        key, metrics = results.query_single_next_metrics(('nvtx_range_name', 'nvtx_push_region_A', 'nvtx_push_region_kernel',))
        assert key == 'kernel'
        assert metrics['smsp__inst_executed.sum'] == 100.

    def test_iter_metrics(self, results : ncu.ProfilingResults) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.ProfilingResults.iter_metrics`.
        """
        [(key, metrics)] = results.iter_metrics(('nvtx_range_name', 'nvtx_push_region_A', 'nvtx_push_region_kernel'))
        assert key == 'kernel'
        assert metrics['smsp__inst_executed.sum'] == 100.

        with pytest.raises(TypeError, match = "Expecting entry 'nvtx_push_region_kernel' to be a leaf node with profiling metrics at ()."):
            next(results.query(('nvtx_range_name', 'nvtx_push_region_A',)).iter_metrics())

    def test_assign_metrics(self) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.ProfilingResults.assign_metrics`.
        """
        other_results = ncu.ProfilingResults()

        other_results.assign_metrics(
            accessors = ('nvtx_range_name', 'push_region_XS', 'nice-kernel'),
            data = {'my-value' : 42},
        )

        assert other_results.query_metrics(accessors = ('nvtx_range_name', 'push_region_XS', 'nice-kernel',))['my-value'] == 42

    def test_aggregate_metrics(self) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.ProfilingResults.aggregate_metrics`.
        """
        other_results = ncu.ProfilingResults()
        other_results.assign_metrics(accessors = ('range-0', 'range-1', 'kernel-A'), data = {'my-counter-int' : 42, 'my-counter-float' : 666.})
        other_results.assign_metrics(accessors = ('range-0', 'range-1', 'kernel-B'), data = {'my-counter-int' : 43, 'my-counter-float' : 667.})
        other_results.assign_metrics(accessors = ('range-0', 'range-1', 'kernel-C'), data = {'my-counter-int' : 44, 'my-counter-float' : 668.})

        aggregated = {'my-counter-int' : 42 + 43 + 44, 'my-counter-float' : 666. + 667. + 668.}

        assert other_results.aggregate_metrics(accessors = ('range-0', 'range-1')) == aggregated

        aggregated.pop('my-counter-float')

        assert other_results.aggregate_metrics(accessors = ('range-0', 'range-1'), keys = ('my-counter-int',)) == aggregated

    def test_string_representation(self, results : ncu.ProfilingResults) -> None:
        """
        Test the string representation of :py:meth:`reprospect.tools.ncu.ProfilingResults`.
        """
        assert str(results) == """\
Profiling results
└── nvtx_range_name
    ├── nvtx_push_region_A
    │   └── nvtx_push_region_kernel
    │       └── kernel
    │           ├── smsp__inst_executed.sum: 100.0
    │           ├── sass__inst_executed_per_opcode: MetricCorrelationData(correlated={'LDCU': 16.0, 'LDC': 16.0}, value=None)
    │           └── L1/TEX cache global load sectors.sum: 0.0
    └── nvtx_push_region_B
        └── nvtx_push_region_other_kernel
            └── other_kernel
                ├── smsp__inst_executed.sum: 96.0
                ├── sass__inst_executed_per_opcode: MetricCorrelationData(correlated={'LDCU': 16.0, 'LDC': 16.0}, value=None)
                └── L1/TEX cache global load sectors.sum: 0.0
"""

        results_A = results.query(("nvtx_range_name", "nvtx_push_region_A"))
        assert isinstance(results_A, ncu.ProfilingResults)
        assert str(results_A) == """\
Profiling results
└── nvtx_push_region_kernel
    └── kernel
        ├── smsp__inst_executed.sum: 100.0
        ├── sass__inst_executed_per_opcode: MetricCorrelationData(correlated={'LDCU': 16.0, 'LDC': 16.0}, value=None)
        └── L1/TEX cache global load sectors.sum: 0.0
"""

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestSession:
    """
    Test :py:class:`reprospect.tools.ncu.Session`.
    """
    GRAPH : typing.Final[pathlib.Path] = pathlib.Path('tests') / 'cpp' / 'cuda' / 'tests_cpp_cuda_graph'
    SAXPY : typing.Final[pathlib.Path] = pathlib.Path('tests') / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy'

    def test_collect_basic_metrics_saxpy_with_nvtx_filtering(self, bindir, workdir) -> None:
        """
        Collect a few basic metrics for the :py:attr:`SAXPY` executable and filter by NVTX.
        """
        METRICS : tuple[ncu.Metric, ...] = (
            # Metric with full name provided.
            ncu.Metric(name = 'launch__registers_per_thread_allocated'),
            # Metric with roll-up.
            ncu.MetricCounter(name = 'smsp__inst_executed', subs = [ncu.MetricCounter.RollUp.SUM]),
            # Launch grid and block.
            *ncu.LaunchBlock.get(),
            *ncu.LaunchGrid.get(),
            # A few L1TEX cache metrics.
            ncu.L1TEXCache.GlobalLoad.Instructions(),
            ncu.L1TEXCache.GlobalLoad.Sectors(),
        )

        EXPT_METRICS_AND_METADATA = (
            'launch__registers_per_thread_allocated',
            'smsp__inst_executed.sum',
            'launch__block_dim_x',
            'launch__block_dim_y',
            'launch__block_dim_z',
            'launch__grid_dim_x',
            'launch__grid_dim_y',
            'launch__grid_dim_z',
            'L1/TEX cache global load instructions sass.sum',
            'L1/TEX cache global load sectors.sum',
            'mangled',
            'demangled',
        )

        session = ncu.Session(output = workdir / 'report-saxpy-basics')

        session.run(
            executable = bindir / self.SAXPY,
            metrics = METRICS,
            nvtx_includes = map(
                lambda x: f'application_domain@{x}/',
                ('launch_saxpy_kernel_first_time', 'launch_saxpy_kernel_second_time'),
            ),
            retries = 5,
        )

        report = ncu.Report(path = workdir, name = 'report-saxpy-basics')

        assert report.report.num_ranges() == 1

        # Extract results without NVTX filtering.
        results = report.extract_results_in_range(0, metrics = METRICS)

        logging.info(results)

        metrics_saxpy_kernel_0 = results.query_metrics(('saxpy_kernel-0',))
        metrics_saxpy_kernel_1 = results.query_metrics(('saxpy_kernel-1',))
        assert all(x in metrics_saxpy_kernel_0 for x in EXPT_METRICS_AND_METADATA)
        assert all(x in metrics_saxpy_kernel_1 for x in EXPT_METRICS_AND_METADATA)
        # Extract results with NVTX filtering. Request only the 2 first metrics.
        with pytest.raises(RuntimeError, match = 'no action found'):
            results_filtered = report.extract_results_in_range(0, metrics = METRICS[:2], includes = ('outer_useless_range',))

        results_filtered = report.extract_results_in_range(0, metrics = METRICS[:2], includes = ('application_domain@outer_useless_range',))

        logging.info(results_filtered)

        results_filtered_first  = results_filtered.query(('launch_saxpy_kernel_first_time',))
        results_filtered_second = results_filtered.query(('launch_saxpy_kernel_second_time',))

        assert isinstance(results_filtered_first,  ncu.ProfilingResults)
        assert isinstance(results_filtered_second, ncu.ProfilingResults)

        metrics_filtered_first_saxpy_kernel_0  = results_filtered_first.query_metrics( ('saxpy_kernel-0',))
        metrics_filtered_second_saxpy_kernel_1 = results_filtered_second.query_metrics(('saxpy_kernel-1',))
        assert all(x in metrics_filtered_first_saxpy_kernel_0  for x in EXPT_METRICS_AND_METADATA[:2])
        assert all(x in metrics_filtered_second_saxpy_kernel_1 for x in EXPT_METRICS_AND_METADATA[:2])

        # A few checks.
        assert metrics_saxpy_kernel_0['launch__block_dim_x'] == 128
        assert metrics_saxpy_kernel_0['launch__block_dim_y'] == 1
        assert metrics_saxpy_kernel_0['launch__block_dim_z'] == 1
        assert metrics_saxpy_kernel_0['launch__grid_dim_x'] == 8
        assert metrics_saxpy_kernel_0['launch__grid_dim_y'] == 1
        assert metrics_saxpy_kernel_0['launch__grid_dim_z'] == 1

        assert metrics_saxpy_kernel_0['mangled'] == '_Z12saxpy_kerneljfPKfPf'
        assert metrics_saxpy_kernel_0['demangled'] == 'saxpy_kernel(unsigned int, float, const float *, float *)'

        for metric in EXPT_METRICS_AND_METADATA[:2]:
            assert metrics_filtered_first_saxpy_kernel_0 [metric] == metrics_saxpy_kernel_0[metric]
            assert metrics_filtered_second_saxpy_kernel_1[metric] == metrics_saxpy_kernel_1[metric]

    def test_collect_correlated_metrics_saxpy(self, bindir, workdir) -> None:
        """
        Collect a metric with correlations for the :py:attr:`SAXPY` executable.
        """
        METRICS = (ncu.MetricCorrelation(name = 'sass__inst_executed_per_opcode'),)

        session = ncu.Session(output = workdir / 'report-saxpy-correlated')

        session.run(executable = bindir / self.SAXPY, metrics = METRICS, retries = 5)

        report = ncu.Report(session = session)

        assert report.report.num_ranges() == 1

        results = report.extract_results_in_range(0, metrics = METRICS)

        logging.info(results)

        metrics_saxpy_kernel_0 = results.query_metrics(('saxpy_kernel-0',))
        assert isinstance(metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'], ncu.MetricCorrelationData)

        # Check that the sum of correlated values matches the total value.
        assert sum(metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'].correlated.values()) == metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'].value

        # Check that the executed instructions include FFMA and IMAD.
        assert metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'].correlated['FFMA'] > 0
        assert metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'].correlated['IMAD'] > 0

        # Check load instructions.
        assert metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'].correlated is not None
        assert 'LD' not in metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'].correlated
        assert metrics_saxpy_kernel_0['sass__inst_executed_per_opcode'].correlated['LDG'] > 0

    def test_collect_basic_metrics_graph(self, bindir, workdir, cmake_cuda_compiler : dict) -> None:
        """
        Collect a few basic metrics for the :py:attr:`GRAPH` executable.
        """
        METRICS = (
            # A few L1TEX cache metrics.
            ncu.L1TEXCache.GlobalLoad.Sectors(),
            ncu.L1TEXCache.GlobalStore.Sectors(),
        )

        session = ncu.Session(output = workdir / 'report-graph-basics')

        session.run(executable = bindir / self.GRAPH, metrics = METRICS, retries = 5)

        report = ncu.Report(session = session)

        assert report.report.num_ranges() == 1

        # Extract results.
        results = report.extract_results_in_range(0, metrics = METRICS, demangler = get_demangler_for_compiler(cmake_cuda_compiler['id']))

        logging.info(results)

        # There are 4 nodes.
        assert len(results) == 4

        match cmake_cuda_compiler['id']:
            case 'NVIDIA':
                NODE_A_MANGLED = '_Z24add_and_increment_kernelILj0EJEEvPj'
                metrics_node_A = results.query_metrics(accessors = ('add_and_increment_kernel-0',))
            case 'Clang':
                # For some reason, ncu cannot demangle the signature of node A when compiling with clang 21.1.3.
                NODE_A_MANGLED = '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj'
                metrics_node_A = results.query_metrics(accessors = (f'{NODE_A_MANGLED}-0',))
            case _:
                raise ValueError(f"unsupported compiler ID {cmake_cuda_compiler['id']}")

        metrics_node_B = results.query_metrics(accessors = ('add_and_increment_kernel-1',))
        metrics_node_C = results.query_metrics(accessors = ('add_and_increment_kernel-2',))
        metrics_node_D = results.query_metrics(accessors = ('add_and_increment_kernel-3',))

        # Check global load/store for each node, and aggregated as well.
        SIGNATURES = {
            'node_A' : lambda x : re.match(r'void add_and_increment_kernel<(?:\(unsigned int\)0, |0u)>\(unsigned int\s*\*\)', x['demangled']),
            'node_B' : lambda x : re.match(r'void add_and_increment_kernel<(?:\(unsigned int\)1, \(unsigned int\)0|1u, 0u)>\(unsigned int\s*\*\)', x['demangled']),
            'node_C' : lambda x : re.match(r'void add_and_increment_kernel<(?:\(unsigned int\)2, \(unsigned int\)0|2u, 0u)>\(unsigned int\s*\*\)', x['demangled']),
            'node_D' : lambda x : re.match(r'void add_and_increment_kernel<(?:\(unsigned int\)3, \(unsigned int\)1, \(unsigned int\)2|3u, 1u, 2u)>\(unsigned int\s*\*\)', x['demangled']),
        }

        assert SIGNATURES['node_A'](metrics_node_A)
        assert SIGNATURES['node_B'](metrics_node_B)
        assert SIGNATURES['node_C'](metrics_node_C)
        assert SIGNATURES['node_D'](metrics_node_D)

        metrics_aggregate = results.aggregate_metrics(accessors = (), keys = (
            'L1/TEX cache global store sectors.sum',
            'L1/TEX cache global load sectors.sum',
        ))

        # Node A makes one load and one store.
        assert metrics_node_A['L1/TEX cache global load sectors.sum']  == 1
        assert metrics_node_A['L1/TEX cache global store sectors.sum'] == 1

        # Nodes B and C make two loads and one store each.
        assert metrics_node_B['L1/TEX cache global load sectors.sum']  == 2
        assert metrics_node_B['L1/TEX cache global store sectors.sum'] == 1
        assert metrics_node_C['L1/TEX cache global load sectors.sum']  == 2
        assert metrics_node_C['L1/TEX cache global store sectors.sum'] == 1

        # Node D makes three loads and one store.
        assert metrics_node_D['L1/TEX cache global load sectors.sum']  == 3
        assert metrics_node_D['L1/TEX cache global store sectors.sum'] == 1

        assert metrics_aggregate == {
            'L1/TEX cache global load sectors.sum':  8,
            'L1/TEX cache global store sectors.sum': 4,
        }

    def test_fails_correctly_with_retries(self, bindir, workdir) -> None:
        """
        When `retries` is given, but `ncu` fails for a good reason, it should not retry, *i.e.* it should not sleep.
        """
        session = ncu.Session(output = workdir / 'report-failure')

        with unittest.mock.patch('time.sleep') as mock_sleep:
            with pytest.raises(subprocess.CalledProcessError):
                session.run(executable = bindir / self.GRAPH, opts = ['--something-ncu-does-not-know'], retries = 5)
            mock_sleep.assert_not_called()

class TestCacher:
    """
    Tests for :py:class:`reprospect.tools.ncu.Cacher`.
    """
    GRAPH : typing.Final[pathlib.Path] = pathlib.Path('tests') / 'cpp' / 'cuda' / 'tests_cpp_cuda_graph'
    SAXPY : typing.Final[pathlib.Path] = pathlib.Path('tests') / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy'

    def test_hash_same(self, bindir) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with ncu.Cacher(directory = tmpdir, session = ncu.Session(output = pathlib.Path('I-dont-care'))) as cacher:
                hash_a = cacher.hash(opts = ['--nvtx'], executable = bindir / self.GRAPH, args = ['--bla=42'])
                hash_b = cacher.hash(opts = ['--nvtx'], executable = bindir / self.GRAPH, args = ['--bla=42'])

                assert hash_a.digest() == hash_b.digest()

    def test_hash_different(self, bindir) -> None:
        """
        Test :py:meth:`reprospect.tools.ncu.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with ncu.Cacher(directory = tmpdir, session = ncu.Session(output = pathlib.Path('I-dont-care'))) as cacher:
                hash_a = cacher.hash(opts = ['--nvtx'], executable = bindir / self.GRAPH, args = ['--bla=42'])
                hash_b = cacher.hash(opts = ['--nvtx'], executable = bindir / self.SAXPY, args = ['--bla=42'])

                assert hash_a.digest() != hash_b.digest()

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_cache_hit(self, bindir, workdir) -> None:
        """
        The cacher should hit on the second call.
        """
        METRICS = (
            # A few L1TEX cache metrics.
            ncu.L1TEXCache.GlobalLoad.Sectors(),
            ncu.L1TEXCache.GlobalStore.Sectors(),
        )

        FILES = ('report-cached.log', 'report-cached.ncu-rep')

        with tempfile.TemporaryDirectory() as tmpdir:
            with ncu.Cacher(directory = tmpdir, session = ncu.Session(output = workdir / 'report-cached')) as cacher:
                assert os.listdir(cacher.directory) == ['cache.db']

                results_first = cacher.run(executable = bindir / self.GRAPH, metrics = METRICS, retries = 5)

                assert results_first.cached is False

                assert all(x in os.listdir(cacher.session.output.parent) for x in FILES)

                for file in FILES:
                    (cacher.session.output.parent / file).unlink()

                assert sorted(os.listdir(cacher.directory)) == sorted(['cache.db', results_first.digest])
                assert sorted(os.listdir(cacher.directory / results_first.digest)) == sorted(FILES)

                results_second = cacher.run(executable = bindir / self.GRAPH, metrics = METRICS, retries = 5)

                assert results_second.cached is True

                assert all(x in os.listdir(cacher.session.output.parent) for x in FILES)
