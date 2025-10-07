import logging
import os
import pathlib
import subprocess
import tempfile
import unittest.mock

import pytest

from reprospect.tools import ncu

TMPDIR = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) if 'CMAKE_CURRENT_BINARY_DIR' in os.environ else None

class TestSession:
    """
    Test :py:class:`reprospect.tools.ncu.Session`.
    """
    GRAPH = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_graph' if 'CMAKE_BINARY_DIR' in os.environ else None
    SAXPY = pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'tests' / 'cpp' / 'cuda' / 'tests_cpp_cuda_saxpy' if 'CMAKE_BINARY_DIR' in os.environ else None

    def test_collect_basic_metrics_saxpy_with_nvtx_filtering(self):
        """
        Collect a few basic metrics for the `saxpy` executable and filter by `NVTX`.
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

        session.run(executable = self.SAXPY, metrics = METRICS, nvtx_capture = 'application-domain@outer-useless-range', retries = 5)

        report = ncu.Report(path = TMPDIR, name = 'report-saxpy-basics')

        assert report.report.num_ranges() == 1

        # Extract results without NVTX filtering.
        results = report.extract_metrics_in_range(0, metrics = METRICS)

        logging.info(results)

        assert len(results) > 1
        assert 'saxpy_kernel-0' in results.keys()
        assert all(x in results['saxpy_kernel-0'] for x in EXPT_METRICS_AND_METADATA)

        # Extract results with NVTX filtering. Request only the 2 first metrics.
        with pytest.raises(RuntimeError, match = 'no action found'):
            results_filtered = report.extract_metrics_in_range(0, metrics = METRICS[:2], includes = ['outer-useless-range'])

        results_filtered = report.extract_metrics_in_range(0, metrics = METRICS[:2], includes = ['application-domain@outer-useless-range'])['launch-saxpy-kernel']

        logging.info(results_filtered)

        assert len(results_filtered) == 1
        assert 'saxpy_kernel-0' in results_filtered.keys()
        assert all(x in results['saxpy_kernel-0'] for x in EXPT_METRICS_AND_METADATA[:2])

        # A few checks.
        assert results['saxpy_kernel-0']['launch__block_dim_x'] == 128
        assert results['saxpy_kernel-0']['launch__block_dim_y'] == 1
        assert results['saxpy_kernel-0']['launch__block_dim_z'] == 1
        assert results['saxpy_kernel-0']['launch__grid_dim_x'] == 8
        assert results['saxpy_kernel-0']['launch__grid_dim_y'] == 1
        assert results['saxpy_kernel-0']['launch__grid_dim_z'] == 1

        assert results['saxpy_kernel-0']['mangled'] == '_Z12saxpy_kerneljfPKfPf'
        assert results['saxpy_kernel-0']['demangled'] == 'saxpy_kernel(unsigned int, float, const float *, float *)'

        for k, v in results_filtered['saxpy_kernel-0'].items():
            assert v == results['saxpy_kernel-0'][k]

    def test_collect_correlated_metrics_saxpy(self):
        """
        Collect a metric with correlations for the `saxpy` executable.
        """
        METRICS = [ncu.MetricCorrelation(name = 'sass__inst_executed_per_opcode')]

        session = ncu.Session(output = TMPDIR / 'report-saxpy-correlated')

        session.run(executable = self.SAXPY, metrics = METRICS)

        report = ncu.Report(session = session)

        assert report.report.num_ranges() == 1

        results = report.extract_metrics_in_range(0, metrics = METRICS)

        logging.info(results)

        assert 'saxpy_kernel-0' in results.keys()

        assert 'sass__inst_executed_per_opcode' in results['saxpy_kernel-0']

        assert sum(results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated.values()) == results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].value

        assert results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated['FFMA'] > 0
        assert results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated['IMAD'] > 0

        # nvcc will use LDG, but clang sticks to LD.
        match os.environ['CMAKE_CUDA_COMPILER_ID']:
            case 'NVIDIA':
                assert 'LD' not in results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated
                assert results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated['LDG'] > 0
            case 'Clang':
                assert 'LDG' not in results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated
                assert results['saxpy_kernel-0']['sass__inst_executed_per_opcode'].correlated['LD'] > 0
            case _:
                raise ValueError(f'unsupported compiler ID {os.environ['CMAKE_CUDA_COMPILER_ID']}')

    def test_collect_basic_metrics_graph(self):
        """
        Collect a few basic metrics for the `graph` executable.
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

        match os.environ['CMAKE_CUDA_COMPILER_ID']:
            case 'NVIDIA':
                NODE_A_MANGLED = '_Z24add_and_increment_kernelILj0EJEEvPj'
                assert all(f'add_and_increment_kernel-{idx}' == k for idx, k in enumerate(results.keys()))
            case 'Clang':
                # For some reason, ncu cannot demangle the signature of node A when compiling with clang 21.1.3.
                NODE_A_MANGLED = '_Z24add_and_increment_kernelILj0ETpTnjJEEvPj'
                assert NODE_A_MANGLED + '-0' in results.keys()
                assert all(f'add_and_increment_kernel-{idx}' in results.keys() for idx in range(1, 4))
            case _:
                raise ValueError(f'unsupported compiler ID {os.environ['CMAKE_CUDA_COMPILER_ID']}')

        # Check global load/store for each node, and aggregated as well.
        # For nvcc:
        #   * node A makes one load and one store.
        #   * nodes B and C make two loads and one store each.
        #   * node D makes three loads and one store.
        # For clang, it is much less efficient, as it performs the "fetch at index and add to other index" each
        # time with a load/store from global memory instead of doing everything at once in a register.
        # Therefore:
        #   * nodes B and C make three loads and two stores each.
        #   * node D makes five loads and three stores.
        match os.environ['CMAKE_CUDA_COMPILER_ID']:
            case 'NVIDIA':
                expt_aggregate = {
                    'L1/TEX cache global load sectors.sum': 8,
                    'L1/TEX cache global store sectors.sum': 4,
                }
            case 'Clang':
                expt_aggregate = {
                    'L1/TEX cache global load sectors.sum': 12,
                    'L1/TEX cache global store sectors.sum': 8,
                }
            case _:
                raise ValueError(f'unsupported compiler ID {os.environ['CMAKE_CUDA_COMPILER_ID']}')

        assert results.aggregate(accessors = [], keys = [
            'L1/TEX cache global store sectors.sum',
            'L1/TEX cache global load sectors.sum',
        ]) == expt_aggregate

        metrics_node_A = next(filter(lambda x: x['mangled'] == NODE_A_MANGLED, results.values()))
        metrics_node_B = next(filter(lambda x: x['mangled'] == '_Z24add_and_increment_kernelILj1EJLj0EEEvPj', results.values()))
        metrics_node_C = next(filter(lambda x: x['mangled'] == '_Z24add_and_increment_kernelILj2EJLj0EEEvPj', results.values()))
        metrics_node_D = next(filter(lambda x: x['mangled'] == '_Z24add_and_increment_kernelILj3EJLj1ELj2EEEvPj', results.values()))

        match os.environ['CMAKE_CUDA_COMPILER_ID']:
            case 'NVIDIA':
                assert metrics_node_A['L1/TEX cache global load sectors.sum'] == 1
                assert metrics_node_A['L1/TEX cache global store sectors.sum'] == 1

                assert metrics_node_B['L1/TEX cache global load sectors.sum'] == 2
                assert metrics_node_B['L1/TEX cache global store sectors.sum'] == 1

                assert metrics_node_C['L1/TEX cache global load sectors.sum'] == 2
                assert metrics_node_C['L1/TEX cache global store sectors.sum'] == 1

                assert metrics_node_D['L1/TEX cache global load sectors.sum'] == 3
                assert metrics_node_D['L1/TEX cache global store sectors.sum'] == 1
            case 'Clang':
                assert metrics_node_A['L1/TEX cache global load sectors.sum'] == 1
                assert metrics_node_A['L1/TEX cache global store sectors.sum'] == 1

                assert metrics_node_B['L1/TEX cache global load sectors.sum'] == 3
                assert metrics_node_B['L1/TEX cache global store sectors.sum'] == 2

                assert metrics_node_C['L1/TEX cache global load sectors.sum'] == 3
                assert metrics_node_C['L1/TEX cache global store sectors.sum'] == 2

                assert metrics_node_D['L1/TEX cache global load sectors.sum'] == 5
                assert metrics_node_D['L1/TEX cache global store sectors.sum'] == 3
            case _:
                raise ValueError(f'unsupported compiler ID {os.environ['CMAKE_CUDA_COMPILER_ID']}')

    def test_fails_correctly_with_retries(self):
        """
        When `retries` is given, but `ncu` fails for a good reason, it should not retry, *i.e.* it should not sleep.
        """
        session = ncu.Session(output = TMPDIR / 'report-failure')

        with unittest.mock.patch('time.sleep') as mock_sleep:
            with pytest.raises(subprocess.CalledProcessError):
                session.run(executable = self.GRAPH, opts = ['--something-ncu-does-not-know'], retries = 5)
            mock_sleep.assert_not_called()

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

                results_first = cacher.run(executable = self.GRAPH, metrics = METRICS)

                assert results_first.cached == False

                assert all(x in os.listdir(cacher.session.output.parent) for x in FILES)

                for file in FILES:
                    (cacher.session.output.parent / file).unlink()

                assert sorted(os.listdir(cacher.directory)) == sorted(['cache.db', results_first.digest])
                assert sorted(os.listdir(cacher.directory / results_first.digest)) == sorted(FILES)

                results_second = cacher.run(executable = self.GRAPH, metrics = METRICS, retries = 5)

                assert results_second.cached == True

                assert all(x in os.listdir(cacher.session.output.parent) for x in FILES)
