import collections
import copy
import dataclasses
import enum
import functools
import importlib
import json
import logging
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
import typing

import blake3
import typeguard

from reprospect.tools import cacher
from reprospect.utils import ldd

class Unit(enum.StrEnum):
    """
    Available units.

    References:
        * https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-decoder
    """
    L1TEX = 'l1tex'
    SM    = 'sm'
    SMSP  = 'smsp'

class PipeStage(enum.StrEnum):
    """
    Available pipe stages.
    """
    TAG = 't'
    TAG_OUTPUT = 't_output'

class Quantity(enum.StrEnum):
    """
    Available quantities.

    References:
        * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder
    """
    INSTRUCTION = 'inst'
    REQUEST     = 'requests'
    SECTOR      = 'sectors'
    WAVEFRONT   = 'wavefronts'

@dataclasses.dataclass(frozen = False)
class Metric:
    """
    Used to represent a `ncu` metric.

    References:
        * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    #: The base name of the metric.
    name : str

    #: Human readable name.
    human : str

    #: A dictionary of sub-metrics and their value.
    subs : dict[typing.Any, int | float]

    @typeguard.typechecked
    def __init__(self,
        name : str,
        *,
        subs : typing.Optional[list[typing.Any]] = None,
        human : typing.Optional[str] = None,
    ) -> None:
        """
        If `subs` is not given, it is assumed that `name` is a valid metric
        that can be directly evaluated by `ncu`.
        """
        self.name  = name
        self.human = human if human else self.name
        self.subs  = {x: None for x in subs} if subs else {'' : None}

    def __repr__(self) -> str:
        return self.name + '(' + ','.join([f'{sub}={value}' for sub, value in self.subs.items()]) + ')'

    @typeguard.typechecked
    def gather(self) -> list[str]:
        """
        Get the list of sub-metric names.
        """
        return [f'{self.name}{sub}' for sub in self.subs.keys()]

class MetricCounter(Metric):
    """
    A counter metric.
    """
    class RollUp(enum.StrEnum):
        SUM = '.sum'
        AVG = '.avg'
        MIN = '.min'
        MAX = '.max'

class MetricRatio(Metric):
    """
    A ratio metric.
    """
    class RollUp(enum.StrEnum):
        PCT      = '.pct'
        RATIO    = '.ratio'
        MAX_RATE = '.max_rate'

class XYZBase:
    """
    Used to represent a triplet of metrics in x, y and z dimensions.

    References:
        * https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-reference
    """
    @classmethod
    @typeguard.typechecked
    def get(cls, dims : list[str] = ['x', 'y', 'z']) -> list[Metric]:
        return [Metric(name = cls.prefix + x) for x in dims]

class LaunchBlock(XYZBase):
    prefix = 'launch__block_dim_'

class LaunchGrid(XYZBase):
    prefix = 'launch__grid_dim_'

@dataclasses.dataclass(frozen = False)
class MetricCorrelation:
    """
    A metric with correlations, like `sass__inst_executed_per_opcode`.
    """
    name : str
    correlated : dict[str, float] = None
    value : float = None

    @typeguard.typechecked
    def gather(self) -> list[str]:
        return [self.name]

@typeguard.typechecked
def counter_name_from(
    *,
    unit : Unit,
    pipestage : typing.Optional[PipeStage] = None,
    quantity : Quantity | str,
    qualifier : typing.Optional[str] = None
) -> str:
    """
    Based on metrics naming convention from https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure::

        unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)
    """
    name = f'{unit}__'
    if pipestage:
        name += f'{pipestage}_'
    name += f'{quantity}'
    if qualifier:
        name += f'_{qualifier}'
    return name

class L1TEXCache:
    """
    A selection of metrics related to `L1/TEX` cache.

    See :cite:`nvidia-ncu-requests-wavefronts-sectors`.
    """
    name = 'L1/TEX cache'

    class GlobalLoad:

        name = 'global load'

        class Instructions(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM], unit : Unit = Unit.SMSP, mode : typing.Optional[typing.Literal['sass']] = 'sass') -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = unit,
                        quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
                        qualifier = 'executed_op_global_ld',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalLoad.name, 'instructions', mode or '']),
                )

        class Requests(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM]) -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = Unit.L1TEX,
                        pipestage = PipeStage.TAG,
                        quantity = Quantity.REQUEST,
                        qualifier = 'pipe_lsu_mem_global_op_ld',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalLoad.name, 'requests']),
                )

        class Sectors(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM]) -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = Unit.L1TEX,
                        pipestage = PipeStage.TAG,
                        quantity = Quantity.SECTOR,
                        qualifier = 'pipe_lsu_mem_global_op_ld',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalLoad.name, 'sectors']),
                )

        class SectorHits(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM]) -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = Unit.L1TEX,
                        pipestage = PipeStage.TAG,
                        quantity = Quantity.SECTOR,
                        qualifier = 'pipe_lsu_mem_global_op_ld_lookup_hit',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalLoad.name, 'sector hits']),
                )

        class SectorMisses(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM]) -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = Unit.L1TEX,
                        pipestage = PipeStage.TAG,
                        quantity = Quantity.SECTOR,
                        qualifier = 'pipe_lsu_mem_global_op_ld_lookup_miss',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalLoad.name, 'sector misses']),
                )

        class Wavefronts(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM]) -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = Unit.L1TEX,
                        pipestage = PipeStage.TAG_OUTPUT,
                        quantity = Quantity.WAVEFRONT,
                        qualifier = 'pipe_lsu_mem_global_op_ld',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalLoad.name, 'wavefronts']),
                )

    class GlobalStore:

        name = 'global store'

        class Instructions(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM], unit : Unit = Unit.SMSP, mode : typing.Optional[typing.Literal['sass']] = 'sass') -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = unit,
                        quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
                        qualifier = 'executed_op_global_st',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalStore.name, 'instructions', mode or '']),
                )

        class Sectors(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM]) -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = Unit.L1TEX,
                        pipestage = PipeStage.TAG,
                        quantity = Quantity.SECTOR,
                        qualifier = 'pipe_lsu_mem_global_op_st',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.GlobalStore.name, 'sectors']),
                )

    class LocalStore:

        name = 'local store'

        class Instructions(MetricCounter):
            @typeguard.typechecked
            def __init__(self, subs = [MetricCounter.RollUp.SUM], unit : Unit = Unit.SMSP, mode : typing.Optional[typing.Literal['sass']] = 'sass') -> None:
                MetricCounter.__init__(self,
                    name  = counter_name_from(
                        unit = unit,
                        quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
                        qualifier = 'executed_op_local_st',
                    ), subs = subs,
                    human = ' '.join([L1TEXCache.name, L1TEXCache.LocalStore.name, 'instructions', mode or '']),
                )

@typeguard.typechecked
def gather(metrics : list[Metric | MetricCorrelation]) -> list[str]:
    """
    Retrieve all sub-metric names, e.g. to pass them to `ncu`.
    """
    result = []
    for metric in metrics:
        result += metric.gather()
    return result

class Session:
    """
    `ncu` session.
    """
    @typeguard.typechecked
    def __init__(self, *, output : pathlib.Path):
        self.output = output

    @dataclasses.dataclass(frozen = True)
    class Command:
        """
        The command is split in:
            * `ncu` options that do not involve paths
            * `ncu` report file
            * `ncu` log file
            * `ncu` metrics
            * executable
            * executable arguments
        """
        opts: list[str]
        output: pathlib.Path
        log: pathlib.Path
        metrics: list[Metric | MetricCorrelation]
        executable: str | pathlib.Path
        args: list[str | pathlib.Path]

        @functools.cached_property
        @typeguard.typechecked
        def to_list(self) -> list[str | pathlib.Path]:
            """
            Build the full `ncu` command.
            """
            cmd = ['ncu']

            if self.opts:
                cmd += self.opts

            cmd += ['--force-overwrite', '-o', self.output]
            cmd += ['--log-file', self.log]

            if self.metrics:
                cmd.append(f'--metrics={",".join(gather(metrics = self.metrics))}')

            cmd.append(self.executable)

            if self.args:
                cmd += self.args

            return cmd

    @typeguard.typechecked
    def get_command(self, *,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_includes : typing.Optional[list[str]] = None,
        metrics : typing.Optional[list[Metric | MetricCorrelation]] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
    ) -> 'Session.Command':
        """
        Create a :py:class:`Session.Command`.
        """
        if not opts: opts = []

        opts += [
            '--print-summary=per-kernel',
            '--warp-sampling-interval=0',
        ]

        if nvtx_includes:
            opts += [
                '--nvtx',
                '--print-nvtx-rename=kernel',
                *[f'--nvtx-include={x}' for x in nvtx_includes],
            ]

        return Session.Command(
            opts = opts,
            output = self.output,
            log = self.output.with_suffix('.log'),
            metrics = metrics,
            executable = executable,
            args = args,
        )

    @typeguard.typechecked
    def run(
        self,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_includes : typing.Optional[list[str]] = None,
        metrics : typing.Optional[list[Metric | MetricCorrelation]] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
        cwd : typing.Optional[pathlib.Path] = None,
        env : typing.Optional[typing.MutableMapping] = None,
        retries : typing.Optional[int] = None,
        sleep : typing.Callable[[int, int], float] = lambda retry, retries: 3. * (1. - retry / retries),
    ) -> 'Session.Command':
        """
        Run `cmd` with `ncu`.

        :param nvtx_includes: Refer to https://docs.nvidia.com/nsight-compute/2023.3/NsightComputeCli/index.html#nvtx-filtering.
        :param retries: `ncu` might fail acquiring some resources because other instances are running. Retry a few times.
                        See https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq (Profiling failed because a driver resource was unavailable).

        :param sleep: The time to sleep between successive retries.
                      The callable is given the current retry index (descending) and the amount of allowed retries.

        .. warning::

            According to https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#ElevPrivsTag,
            `GPU` performance counters are not available to all users by default.

        .. note::

            As of `ncu` `2025.1.1.0`, a note tells us that specified `NVTX` include expressions match only start/end ranges.

        References:
            * https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering
        """
        command = self.get_command(
            opts = opts,
            nvtx_includes = nvtx_includes,
            metrics = metrics,
            executable = executable,
            args = args,
        )

        for retry in reversed(range(retries if retries else 1)):
            try:
                logging.info(f"Launching 'ncu' with {command.to_list} (log file at {command.log}).")
                subprocess.check_call(command.to_list, cwd = cwd, env = env)
                break
            except subprocess.CalledProcessError:
                retry_allowed = False
                if retry > 0 and command.log.is_file():
                    with open(self.output.with_suffix('.log'), 'r') as fin:
                        for line in fin:
                            if line.startswith('==ERROR== Profiling failed because a driver resource was unavailable.'):
                                logging.warning(f'Retrying because a driver resource was unavailable.')
                                retry_allowed = True
                                break

                if not retry_allowed:
                    logging.exception(f'Failed launching \'ncu\' with {command}.{'\n' + command.log.read_text() if command.log.is_file() else ''}')
                    raise
                else:
                    sleep_for = sleep(retry, retries)
                    logging.info(f'Sleeping {sleep_for} seconds before retrying.')
                    time.sleep(sleep_for)

        return command

class Range:
    """
    Wrapper around :ncu_report:`IRange`.

    This class loads the actions that are in the range.
    """
    @typeguard.typechecked
    def __init__(self, report, index : int, includes : typing.Optional[list[str]] = None, excludes : typing.Optional[list[str]] = None) -> None:
        self.index = index
        self.range = report.range_by_idx(self.index)

        self.actions : list[Action] = []

        self._load_actions_by_nvtx(includes = includes, excludes = excludes)

    @typeguard.typechecked
    def _load_actions_by_nvtx(self, includes : typing.Optional[list[str]] = None, excludes : typing.Optional[list[str]] = None) -> None:
        """
        If both `includes` and `excludes` are empty, load all actions.
        """
        if not includes and not excludes:
            for iaction in range(self.range.num_actions()):
                self.actions.append(Action(range = self.range, index = iaction))
        else:
            for iaction in self.range.actions_by_nvtx(includes if includes else [], excludes if excludes else []):
                self.actions.append(Action(range = self.range, index = iaction))

    def __repr__(self) -> str:
        return f'{self.range} (index {self.index})'

class Action:
    """
    Wrapper around :ncu_report:`IAction`.
    """
    @typeguard.typechecked
    def __init__(self, range, index : int) -> None:
        self.index  = index
        self.action = range.action_by_idx(index)

        self.nvtx_state = self.action.nvtx_state()

        self.domains : list[NvtxDomain] = []

        self._load_domains()

    def _load_domains(self):
        if self.nvtx_state:
            for idomain in self.nvtx_state.domains():
                self.domains.append(NvtxDomain(nvtx_state = self.nvtx_state, index = idomain))

    def __repr__(self) -> str:
        return f"{self.action} (index {self.index}, domains {self.domains}, mangled {self.action.name(self.action.NameBase_MANGLED)})"

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.action, attr)
        raise AttributeError(attr)

class NvtxDomain:
    """
    Wrapper around :ncu_report:`INvtxDomainInfo`.
    """
    def __init__(self, nvtx_state, index : int) -> None:
        self.index = index
        self.nvtx_domain = nvtx_state.domain_by_id(self.index)

    def __repr__(self) -> str:
        return f"{self.nvtx_domain} (index {self.index}, name {self.nvtx_domain.name()} {self.nvtx_domain.start_end_ranges()} {self.nvtx_domain.push_pop_ranges()})"

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.nvtx_domain, attr)
        raise AttributeError(attr)

class ProfilingResults(collections.UserDict):
    """
    Data structure for storing profiling results.

    This data structure is a nested (ordered) dictionary.
    It has helper functions that ease dealing with nested values.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.data = collections.OrderedDict()

    @typeguard.typechecked
    def get(self, accessors : typing.Iterable) -> dict:
        """
        Get (nested) value from the `accessors` path.
        It is created if needed.
        """
        current = self.data
        for accessor in accessors:
            if accessor not in current:
                current[accessor] = {}
            current = current[accessor]
        return current

    @typeguard.typechecked
    def aggregate(self, accessors : typing.Iterable, keys : typing.Optional[typing.Iterable] = None) -> dict:
        """
        Aggregate values of dictionaries selected by `accessors`.

        The selected dictionaries are assumed to be at the last nesting level and to all have the same keys.
        """
        # Get all dictionaries that match 'accessors'.
        samples = self.get(accessors = accessors)

        # Create the aggregate results.
        # Get keys of the first sample if no keys provided. We assume all samples have the same keys.
        aggregate = dict.fromkeys(
            samples[next(iter(samples.keys()))].keys() if keys is None else keys
        )

        for key in aggregate.keys():
            aggregate[key] = sum(map(lambda x: x[key], samples.values()))

        return aggregate

@typeguard.typechecked
def load_ncu_report() -> typing.Optional[types.ModuleType]:
    """
    Attempt to load the Nsight Compute `Python` interface (`ncu_report`).

    Priority:
        1. If `ncu_report` is already imported, return it.
        2. Try a regular `import ncu_report` (for users who set `PYTHONPATH`).
        3. Try to locate `ncu` via `shutil.which`, deduce version, and load
           the bundled `Python` module from Nsight Compute installation path.
    """
    if 'ncu_report' in sys.modules:
        return sys.modules['ncu_report']

    try:
        import ncu_report
        return sys.modules['ncu_report']
    except ImportError:
        pass

    ncu = shutil.which('ncu')
    if ncu is None:
        raise RuntimeError("'ncu' was not found.")

    version_string = subprocess.check_output([ncu, '--version']).decode().splitlines()[-1]
    version_full   = re.match(r'Version ([0-9.]+)', version_string).groups()[0]
    version_short  = re.match(r'[0-9]+.[0-9]+.[0-9]+', version_full).group()
    logging.debug(f'Found \'ncu\' ({ncu}) with version {version_full}.')

    nsight_compute_dir = pathlib.Path('/opt') / 'nvidia' / 'nsight-compute' / version_short
    if not nsight_compute_dir.is_dir():
        raise FileExistsError(f'Presumed Nsight Compute install directory {nsight_compute_dir} does not exist.')

    nsight_compute_extra_python_dir = nsight_compute_dir / 'extras' / 'python'
    ncu_report_spec = importlib.util.spec_from_file_location(
        name                       = "ncu_report",
        location                   = nsight_compute_extra_python_dir / 'ncu_report.py',
        submodule_search_locations = [str(nsight_compute_extra_python_dir)],
    )
    ncu_report = importlib.util.module_from_spec(ncu_report_spec)
    sys.modules['ncu_report'] = ncu_report
    ncu_report_spec.loader.exec_module(ncu_report)
    return sys.modules['ncu_report']

class Report:
    """
    This class is a wrapper around the `Python` tool provided by Nvidia Nsight Compute
    to parse its reports.

    References:
        * https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#python-report-interface
        * https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    @typeguard.typechecked
    def __init__(self, *, path : typing.Optional[pathlib.Path] = None, name : typing.Optional[str] = None, session : typing.Optional[Session] = None) -> None:
        """
        Load the report `<path>/<name>.ncu-rep` or the report generated by `session`.
        """
        self.ncu_report = load_ncu_report()

        # Load the report.
        if path is not None and name is not None and session is None:
            self.path = path / (name + '.ncu-rep')
            if not self.path.is_file():
                raise FileNotFoundError(f"The report path {self.path} is not a file.")
        elif path is None and name is None and session is not None:
            self.path = session.output.with_suffix('.ncu-rep')
        else:
            raise RuntimeError()

        self.report = self.ncu_report.load_report(self.path)

        logging.info(f"Report {self.path} loaded successfully ({self.report}).")

    @typeguard.typechecked
    def extract_metrics_in_range(
        self,
        range_idx : int,
        metrics : list[Metric | MetricCorrelation],
        includes : typing.Optional[list[str]] = None,
        excludes : typing.Optional[list[str]] = None,
    ) -> ProfilingResults:
        """
        Extract the `metrics` of the actions in the range with ID `range_idx`.
        Possibly filter by `NVTX` with `includes` and `excludes`.
        """
        logging.info(f'Collecting metrics {metrics} in report {self.path} for range {range_idx} with include filters {includes} and exclude filters {excludes}.')

        current_range = Range(report = self.report, index = range_idx, includes = includes, excludes = excludes)

        logging.debug(f'Parsing range {current_range}.')

        profiling_results = ProfilingResults()

        if not current_range.actions:
            raise RuntimeError('no action found')

        # Loop over the actions.
        for action in current_range.actions:

            logging.debug(f'Parsing action {action.name()} ({action}).')

            results = self.collect_metrics_from_action(action = action, metrics = metrics)

            results['mangled']   = action.name(action.NameBase_MANGLED)
            results['demangled'] = action.name(action.NameBase_DEMANGLED)

            # Loop over the domains of the action.
            # Note that domains are only available if NVTX was enabled during collection.
            if action.domains:
                for domain in action.domains:
                    current = profiling_results.get(accessors = domain.push_pop_ranges())
                    current[f'{action.name()}-{action.index}'] = results
            else:
                profiling_results[f'{action.name()}-{action.index}'] = results

        return profiling_results

    @typeguard.typechecked
    def collect_metrics_from_action(self, *, metrics : list[Metric | MetricCorrelation], action : Action) -> dict[str, typing.Any]:
        """
        Collect values of the `metrics` in the `action`.

        References:
            * https://github.com/shunting314/gpumisc/blob/37bbb827ae2ed6f5777daff06956c7a10aafe34d/ncu-related/official-sections/FPInstructions.py#L51
            * https://github.com/NVIDIA/nsight-training/blob/2d680f7f8368b945bc00b22834808af24eff4c3d/cuda/nsight_compute/python_report_interface/Opcode_instanced_metrics.ipynb
        """
        results = {}

        for metric in map(copy.deepcopy, metrics):
            if isinstance(metric, MetricCorrelation):
                metric_correlation = action.metric_by_name(metric.name)

                # Recent 'ncu' (>= 2025.3.0.0) provide a 'value' method.
                if hasattr(self.ncu_report, 'IMetric_kind_to_value_func'):
                    metric.value = self.ncu_report.IMetric_kind_to_value_func[metric_correlation.kind()](metric_correlation)
                else:
                    metric.value = metric_correlation.value()

                metric.correlated = {}
                correlations      = metric_correlation.correlation_ids()
                assert correlations.num_instances() == metric_correlation.num_instances()
                for icor in range(metric_correlation.num_instances()):
                    metric.correlated[correlations.as_string(icor)] = metric_correlation.value(icor)
                results[metric.name] = metric
            elif isinstance(metric, Metric):
                self.fill_metric(action = action, metric = metric)
                for sub_name, sub_value in metric.subs.items():
                    results[f'{metric.human}{sub_name}'] = sub_value
            else:
                raise NotImplementedError(metric)

        return results

    @classmethod
    @typeguard.typechecked
    def fill_metric(cls, action : Action, metric : Metric) -> Metric:
        """
        Loop over submetrics of `metric`.
        """
        for sub in metric.subs:
            metric.subs[sub] = cls.get_metric_by_name(action = action, metric = metric.name + sub).value()
        return metric

    @classmethod
    @typeguard.typechecked
    def get_metric_by_name(cls, *, action : Action, metric : str):
        """
        Read a `metric` in `action`.
        """
        collected = action.metric_by_name(metric)
        if collected is None:
            raise RuntimeError(f"There was a problem retrieving metric '{metric}'. It probably does not exist.")
        return collected

class Cacher(cacher.Cacher):
    """
    Cacher tailored to `ncu` results.

    `ncu` require quite some time to acquire results, especially when there are many kernels to profile and/or
    many metrics to collect.

    On a cache hit, the cacher will serve:
        - `.ncu-rep` file
        - `.log` file

    On a cache miss, `ncu` is launched and the cache entry populated accordingly.

    .. note::

        It is assumed that hashing is faster than running `ncu` itself.

    .. warning::

        The cache should not be shared between machines, since there may be differences between machines
        that influence the results but are not included in the hashing.
    """
    TABLE : str = 'ncu'

    @typeguard.typechecked
    def __init__(self, *, session : Session, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    @typing.override
    @typeguard.typechecked
    def hash(self, *,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_includes : typing.Optional[list[str]] = None,
        metrics : typing.Optional[list[Metric | MetricCorrelation]] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
        env : typing.Optional[typing.MutableMapping] = None,
        **kwargs
    ) -> blake3.blake3:
        """
        Hash based on:
            * `ncu` version
            * `ncu` options (but not the output and log files)
            * `ncu` metrics
            * executable content
            * executable arguments
            * linked libraries
            * environment
        """
        hasher = blake3.blake3()

        hasher.update(subprocess.check_output(['ncu', '--version']))

        command = self.session.get_command(
            opts = opts,
            nvtx_includes = nvtx_includes,
            metrics = metrics,
            executable = executable,
            args = args,
        )

        if command.opts:
            hasher.update(shlex.join(command.opts).encode())

        if command.metrics:
            hasher.update(''.join(gather(metrics = command.metrics)).encode())

        hasher.update_mmap(command.executable)

        if command.args:
            hasher.update(shlex.join(command.args).encode())

        for lib in ldd.get_shared_dependencies(file = command.executable):
            hasher.update_mmap(lib)

        if env:
            hasher.update(json.dumps(env).encode())

        return hasher

    @typeguard.typechecked
    def populate(self, directory : pathlib.Path, **kwargs) -> Session.Command:
        """
        When there is a cache miss, call :py:meth:`reprospect.tools.ncu.Session.run`.
        Fill the `directory` with the artifacts.
        """
        command = self.session.run(**kwargs)

        shutil.copy(dst = directory, src = command.output.with_suffix('.ncu-rep'))
        shutil.copy(dst = directory, src = command.log)

        return command

    @typeguard.typechecked
    def run(self, **kwargs) -> cacher.Cacher.Entry:
        """
        On a cache hit, copy files from the cache entry.
        """
        entry = self.get(**kwargs)

        if entry.cached:
            shutil.copytree(entry.directory, self.session.output.parent, dirs_exist_ok = True)

        return entry
