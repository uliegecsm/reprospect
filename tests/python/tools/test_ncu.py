import logging
import os
import pathlib
import subprocess
import tempfile
import unittest.mock

import pytest
import typeguard

from reprospect.utils import cmake
from reprospect.tools import ncu
from reprospect.utils import detect

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

@pytest.fixture(scope = 'session')
@typeguard.typechecked
def cmake_cuda_compiler() -> dict:
    return cmake.FileAPI(
        cmake_build_directory = pathlib.Path(os.environ['CMAKE_BINARY_DIR'])
    ).toolchains['CUDA']['compiler']

@pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
class TestSession:
    """
    Test :py:class:`reprospect.tools.ncu.Session`.
    """
    GRAPH = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_graph' if 'CMAKE_BINARY_DIR' in os.environ else None
    SAXPY = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy' if 'CMAKE_BINARY_DIR' in os.environ else None

    def test_collect_basic_metrics_saxpy_with_nvtx_filtering(self):
        """
        Collect a few basic metrics for the :py:attr:`SAXPY` executable and filter by NVTX.
        """
        METRICS = [
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
        ]

        EXPT_METRICS_AND_METADATA = [
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
        ]

        session = ncu.Session(output = TMPDIR / 'report-saxpy-basics')

        session.run(
            executable = self.SAXPY,
            metrics = METRICS,
            nvtx_includes = map(
                lambda x: f'application_domain@{x}/',
                ['launch_saxpy_kernel_first_time', 'launch_saxpy_kernel_second_time'],
            ),
            retries = 5,
        )

        report = ncu.Report(path = TMPDIR, name = 'report-saxpy-basics')

        assert report.report.num_ranges() == 1

        # Extract results without NVTX filtering.
        results = report.extract_metrics_in_range(0, metrics = METRICS)

        logging.info(results)

        assert len(results) > 1
        assert 'saxpy_kernel-0' in results.keys()
        assert 'saxpy_kernel-1' in results.keys()
        assert all(x in results['saxpy_kernel-0'] for x in EXPT_METRICS_AND_METADATA)
        assert all(x in results['saxpy_kernel-1'] for x in EXPT_METRICS_AND_METADATA)

        # Extract results with NVTX filtering. Request only the 2 first metrics.
        with pytest.raises(RuntimeError, match = 'no action found'):
            results_filtered = report.extract_metrics_in_range(0, metrics = METRICS[:2], includes = ['outer_useless_range'])

        results_filtered = report.extract_metrics_in_range(0, metrics = METRICS[:2], includes = ['application_domain@outer_useless_range'])

        logging.info(results_filtered)

        results_filtered_first  = results_filtered['launch_saxpy_kernel_first_time']
        results_filtered_second = results_filtered['launch_saxpy_kernel_second_time']

        assert len(results_filtered_first)  == 1
        assert len(results_filtered_second) == 1
        assert 'saxpy_kernel-0' in results_filtered_first .keys()
        assert 'saxpy_kernel-1' in results_filtered_second.keys()
        assert all(x in results_filtered_first ['saxpy_kernel-0'] for x in EXPT_METRICS_AND_METADATA[:2])
        assert all(x in results_filtered_second['saxpy_kernel-1'] for x in EXPT_METRICS_AND_METADATA[:2])

        # A few checks.
        assert results['saxpy_kernel-0']['launch__block_dim_x'] == 128
        assert results['saxpy_kernel-0']['launch__block_dim_y'] == 1
        assert results['saxpy_kernel-0']['launch__block_dim_z'] == 1
        assert results['saxpy_kernel-0']['launch__grid_dim_x'] == 8
        assert results['saxpy_kernel-0']['launch__grid_dim_y'] == 1
        assert results['saxpy_kernel-0']['launch__grid_dim_z'] == 1

        assert results['saxpy_kernel-0']['mangled'] == '_Z12saxpy_kerneljfPKfPf'
        assert results['saxpy_kernel-0']['demangled'] == 'saxpy_kernel(unsigned int, float, const float *, float *)'

        for metric in EXPT_METRICS_AND_METADATA[:2]:
            assert results_filtered_first ['saxpy_kernel-0'][metric] == results['saxpy_kernel-0'][metric]
            assert results_filtered_second['saxpy_kernel-1'][metric] == results['saxpy_kernel-1'][metric]

    def test_collect_correlated_metrics_saxpy(self):
        """
        Collect a metric with correlations for the :py:attr:`SAXPY` executable.
        """
        METRICS = [ncu.MetricCorrelation(name = 'sass__inst_executed_per_opcode')]

        session = ncu.Session(output = TMPDIR / 'report-saxpy-correlated')

        session.run(executable = self.SAXPY, metrics = METRICS, retries = 5)

        report = ncu.Report(session = session)

        assert report.report.num_ranges() == 1

        results = report.extract_metrics_in_range(0, metrics = METRICS)

        logging.info(results)

        assert 'saxpy_kernel-0' in results.keys()

        assert 'sass__inst_executed_per_opcode' in results['saxpy_kernel-0']

        assert sum(results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated.values()) == results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].value

        # Check that the executed instructions include FFMA and IMAD.
        assert results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated['FFMA'] > 0
        assert results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated['IMAD'] > 0

        # Check load instructions.
        assert 'LD' not in results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated
        assert results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated['LDG'] > 0

    def test_collect_basic_metrics_graph(self, cmake_cuda_compiler : dict):
        """
        Collect a few basic metrics for the :py:attr:`GRAPH` executable.
        """
        METRICS = [
            # A few L1TEX cache metrics.
            ncu.L1TEXCache.GlobalLoad.Sectors(),
            ncu.L1TEXCache.GlobalStore.Sectors(),
        ]

        session = ncu.Session(output = TMPDIR / 'report-graph-basics')

        session.run(executable = self.GRAPH, metrics = METRICS, retries = 5)

        report = ncu.Report(session = session)

        assert report.report.num_ranges() == 1

        # Extract results.
        results = report.extract_metrics_in_range(0, metrics = METRICS)

        logging.info(results)

        # There are 4 nodes.
        assert len(results) == 4

        match cmake_cuda_compiler['id']:
            case 'NVIDIA':
                NODE_A_MANGLED = '_Z24add_and_increment_kernelILj0EJEEvPj'
                assert all(f'add_and_increment_kernel-{idx}' == k for idx, k in enumerate(results.keys()))
            case 'Clang':
                # For some reason, ncu cannot demangle the signature of node A when compiling with clang 21.1.3.
                NODE_A_MANGLED = '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj'
                assert NODE_A_MANGLED + '-0' in results.keys()
                assert all(f'add_and_increment_kernel-{idx}' in results.keys() for idx in range(1, 4))
            case _:
                raise ValueError(f'unsupported compiler ID {cmake_cuda_compiler['id']}')

        # Check global load/store for each node, and aggregated as well.
        metrics_node_A = next(filter(lambda x: x['mangled'] == NODE_A_MANGLED, results.values()))
        metrics_node_B = next(filter(lambda x: x['mangled'] == '_Z24add_and_increment_kernelILj1EJLj0EEEvPj', results.values()))
        metrics_node_C = next(filter(lambda x: x['mangled'] == '_Z24add_and_increment_kernelILj2EJLj0EEEvPj', results.values()))
        metrics_node_D = next(filter(lambda x: x['mangled'] == '_Z24add_and_increment_kernelILj3EJLj1ELj2EEEvPj', results.values()))

        metrics_aggregate = results.aggregate(accessors = [], keys = [
            'L1/TEX cache global store sectors.sum',
            'L1/TEX cache global load sectors.sum',
        ])

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

    def test_fails_correctly_with_retries(self):
        """
        When `retries` is given, but `ncu` fails for a good reason, it should not retry, *i.e.* it should not sleep.
        """
        session = ncu.Session(output = TMPDIR / 'report-failure')

        with unittest.mock.patch('time.sleep') as mock_sleep:
            with pytest.raises(subprocess.CalledProcessError):
                session.run(executable = self.GRAPH, opts = ['--something-ncu-does-not-know'], retries = 5)
            mock_sleep.assert_not_called()

class TestProfilingResults:
    """
    Tests for :py:class:`reprospect.tools.ncu.ProfilingResults`.
    """
    def test_string_representation(self):
        """
        Test the string representation of :py:meth:`reprospect.tools.ncu.ProfilingResults`.
        """
        results = ncu.ProfilingResults()

        # Add nested data.
        results.get(["nvtx_range_name", "global_function_name_a_idx_0"])["metric_i"] = 15
        results.get(["nvtx_range_name", "global_function_name_a_idx_0"])["metric_ii"] = 7
        results.get(["nvtx_range_name", "global_function_name_b_idx_1"])["metric_i"] = 15
        results.get(["nvtx_range_name", "global_function_name_b_idx_1"])["metric_ii"] = 9

        assert str(results) == """\
Profiling results
└── nvtx_range_name
    ├── global_function_name_a_idx_0
    │   ├── metric_i: 15
    │   └── metric_ii: 7
    └── global_function_name_b_idx_1
        ├── metric_i: 15
        └── metric_ii: 9
"""

class TestCacher:
    """
    Tests for :py:class:`reprospect.tools.ncu.Cacher`.
    """
    GRAPH = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_graph' if 'CMAKE_BINARY_DIR' in os.environ else None
    SAXPY = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy' if 'CMAKE_BINARY_DIR' in os.environ else None

    def test_hash_same(self):
        """
        Test :py:meth:`reprospect.tools.ncu.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with ncu.Cacher(directory = tmpdir, session = ncu.Session(output = pathlib.Path('I-dont-care'))) as cacher:
                hash_a = cacher.hash(opts = ['--nvtx'], executable = self.GRAPH, args = ['--bla=42'])
                hash_b = cacher.hash(opts = ['--nvtx'], executable = self.GRAPH, args = ['--bla=42'])

                assert hash_a.digest() == hash_b.digest()

    def test_hash_different(self):
        """
        Test :py:meth:`reprospect.tools.ncu.Cacher.hash`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with ncu.Cacher(directory = tmpdir, session = ncu.Session(output = pathlib.Path('I-dont-care'))) as cacher:
                hash_a = cacher.hash(opts = ['--nvtx'], executable = self.GRAPH, args = ['--bla=42'])
                hash_b = cacher.hash(opts = ['--nvtx'], executable = self.SAXPY, args = ['--bla=42'])

                assert hash_a.digest() != hash_b.digest()

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_cache_hit(self):
        """
        The cacher should hit on the second call.
        """
        METRICS = [
            # A few L1TEX cache metrics.
            ncu.L1TEXCache.GlobalLoad.Sectors(),
            ncu.L1TEXCache.GlobalStore.Sectors(),
        ]

        FILES = ['report-cached.log', 'report-cached.ncu-rep']

        with tempfile.TemporaryDirectory() as tmpdir:
            with ncu.Cacher(directory = tmpdir, session = ncu.Session(output = TMPDIR / 'report-cached')) as cacher:
                assert os.listdir(cacher.directory) == ['cache.db']

                results_first = cacher.run(executable = self.GRAPH, metrics = METRICS, retries = 5)

                assert results_first.cached is False

                assert all(x in os.listdir(cacher.session.output.parent) for x in FILES)

                for file in FILES:
                    (cacher.session.output.parent / file).unlink()

                assert sorted(os.listdir(cacher.directory)) == sorted(['cache.db', results_first.digest])
                assert sorted(os.listdir(cacher.directory / results_first.digest)) == sorted(FILES)

                results_second = cacher.run(executable = self.GRAPH, metrics = METRICS, retries = 5)

                assert results_second.cached is True

                assert all(x in os.listdir(cacher.session.output.parent) for x in FILES)
