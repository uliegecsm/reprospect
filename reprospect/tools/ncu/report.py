from __future__ import annotations

import collections.abc
import dataclasses
import importlib
import logging
import pathlib
import re
import shutil
import subprocess
import sys
import types
import typing

import rich.tree

from reprospect.tools.binaries.demangle import CuppFilt, LlvmCppFilt
from reprospect.tools.ncu.metrics import (
    Metric,
    MetricCorrelation,
    MetricCorrelationData,
    MetricCorrelationDataType,
    MetricData,
    MetricDeviceAttribute,
    MetricKind,
    ValueType,
)
from reprospect.tools.ncu.session import Command
from reprospect.utils import rich_helpers

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

#: Metric data in profiling results.
ProfilingMetricData: typing.TypeAlias = MetricData | MetricCorrelationData | str

class ProfilingMetrics(collections.abc.Mapping[str, ProfilingMetricData]):
    """
    Mapping of profiling metric keys to their values.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('data',)

    def __init__(self, data: dict[str, ProfilingMetricData]) -> None:
        self.data: typing.Final[dict[str, ProfilingMetricData]] = data

    def __getitem__(self, key: str, /) -> ProfilingMetricData:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.data)

class ProfilingResults(rich_helpers.TreeMixin):
    """
    Nested tree data structure for storing profiling results.

    The data structure consists of internal nodes and leaf nodes:

    * The internal nodes are themselves :py:class:`ProfilingResults` instances. They
      organise results hierarchically by the profiling range (e.g. NVTX range)
      that they were obtained from, terminating by the kernel name.
    * The leaf nodes contain the actual profiling metric key-value pairs. Any type implementing
      the protocol :py:class:`ProfilingMetrics` can be used as a profiling metrics entry.

    This class provides convenient methods for hierarchical data access and manipulation.

    Example structure::

        Profiling results
        └── 'nvtx range'
            ├── 'nvtx region'
            │   └── 'kernel'
            │       ├── 'metric i'  -> ProfilingMetricData
            │       └── 'metric ii' -> ProfilingMetricData
            └── 'other nvtx region'
                └── 'other kernel'
                    ├── 'metric i'  -> ProfilingMetricData
                    └── 'metric ii' -> ProfilingMetricData

    .. note::

        Using a hierarchical data structure from a package such as `hatchet`_
        could be a direction to explore in the future.

    .. _hatchet: https://hatchet.readthedocs.io/

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('data',)

    def __init__(self, data: dict[str, ProfilingResults | ProfilingMetrics] | None = None) -> None:
        self.data: typing.Final[dict[str, ProfilingResults | ProfilingMetrics]] = data if data is not None else {}

    def query(self, accessors: typing.Iterable[str]) -> ProfilingResults | ProfilingMetrics:
        """
        Get the internal node in the hierarchy or the leaf node with profiling metrics
        at accessor path `accessors`.
        """
        current: ProfilingResults | ProfilingMetrics = self
        for accessor in accessors:
            if not isinstance(current, ProfilingResults):
                raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r} instead.')
            current = current.data[accessor]
        return current

    def query_filter(self, accessors: typing.Iterable[str], predicate: typing.Callable[[str], bool]) -> ProfilingResults:
        """
        Query the accessor path `accessors`, check that it leads to an internal node,
        and return a new :py:class:`ProfilingResults` with only the entries
        whose key satisfies `predicate`.
        """
        current = self.query(accessors=accessors)
        if not isinstance(current, ProfilingResults):
            raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r} instead.')
        return ProfilingResults(data={k: v for k, v in current.data.items() if predicate(k)})

    def query_metrics(self, accessors: typing.Iterable[str]) -> ProfilingMetrics:
        """
        Query the accessor path `accessors`, check that it leads to a leaf node with profiling metrics,
        and return this leaf node with profiling metrics.
        """
        current = self.query(accessors=accessors)
        if not isinstance(current, ProfilingMetrics):
            raise TypeError(f'Expecting leaf node with profiling metrics at {accessors}, got {type(current).__name__!r} instead.')
        return current

    def query_single_next(self, accessors: typing.Iterable[str]) -> tuple[str, ProfilingResults | ProfilingMetrics]:
        """
        Query the accessor path `accessors`, check that it leads to an internal node with exactly
        one entry, and return this single entry.

        This member function is useful for instance to access a single kernel
        within an NVTX range or region.

        >>> from reprospect.tools.ncu import ProfilingResults
        >>> results = ProfilingResults()
        >>> results.assign_metrics(('my_nvtx_range', 'my_nvtx_region', 'my_kernel'), {'my_metric' : 42})
        >>> results.query_single_next(('my_nvtx_range', 'my_nvtx_region'))
        ('my_kernel', {'my_metric': 42})
        """
        current = self.query(accessors=accessors)
        if isinstance(current, ProfilingResults):
            if len(current) != 1:
                raise RuntimeError(f'Expecting a single entry at {accessors}, got {len(current)} entries instead.')
            return next(iter(current.data.items()))
        raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r} instead.')

    def query_single_next_metrics(self, accessors: typing.Iterable[str]) -> tuple[str, ProfilingMetrics]:
        """
        Query the accessor path `accessors`, check that it leads to an internal node with exactly
        one entry, check that this entry is a leaf node with profiling metrics, and return this
        leaf node with profiling metrics.
        """
        key, value = self.query_single_next(accessors=accessors)
        if not isinstance(value, ProfilingMetrics):
            raise TypeError(f'Expecting leaf node {key!r} with profiling metrics as the single entry at {accessors}, got {type(value).__name__!r} instead.')
        return key, value

    def iter_metrics(self, accessors: typing.Iterable[str] = ()) -> typing.Generator[tuple[str, ProfilingMetrics], None, None]:
        """
        Query the accessor path `accessors`, check that it leads to an internal node, check that all entries
        are leaf nodes with profiling metrics, and return an iterator over these leaf nodes with profiling metrics.
        """
        current = self.query(accessors=accessors)
        if not isinstance(current, ProfilingResults):
            raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r} instead.')
        for key, value in current.data.items():
            if not isinstance(value, ProfilingMetrics):
                raise TypeError(f'Expecting entry {key!r} to be a leaf node with profiling metrics at {accessors}.')
            yield key, value

    def assign_metrics(self, accessors: typing.Sequence[str], data: ProfilingMetrics) -> None:
        """
        Set the leaf node with profiling metrics `data` at accessor path `accessors`.

        Creates the internal nodes in the hierarchy if needed.
        """
        current = self
        for accessor in accessors[0:-1]:
            value = current.data.setdefault(accessor, ProfilingResults())
            if not isinstance(value, ProfilingResults):
                raise TypeError(f'Expecting internal node at {accessor}, got {type(value).__name__} instead.')
            current = value
        current.data[accessors[-1]] = data

    def aggregate_metrics(self, accessors: typing.Iterable[str], keys: typing.Iterable[str] | None = None) -> dict[str, int | float]:
        """
        Aggregate metric values across multiple leaf nodes with profiling metrics at accessor path `accessors`.

        :param keys: Specific metric keys to aggregate. If :py:obj:`None`, uses all keys from the first leaf node.
        """
        current = self.query(accessors=accessors)

        if not isinstance(current, ProfilingResults):
            raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r}.')

        if keys is None:
            value = next(iter(current.data.values()))
            assert isinstance(value, ProfilingMetrics)
            keys = value.keys()

        return {
            key: sum(
                v for _, m in current.iter_metrics(accessors=())
                if isinstance(v := m[key], int | float)
            )
            for key in keys
        }

    def __len__(self) -> int:
        """
        Get the number of internal nodes.
        """
        return len(self.data)

    @override
    def to_tree(self) -> rich.tree.Tree:
        """
        Convert to a :py:class:`rich.tree.Tree` for nice printing.
        """
        rt = rich.tree.Tree('Profiling results')
        def add_branch(*, tree: rich.tree.Tree, data: ProfilingResults | ProfilingMetrics) -> None:
            for key, value in data.data.items() if isinstance(data, ProfilingResults) else data.items():
                if isinstance(value, ProfilingResults | ProfilingMetrics):
                    branch = tree.add(str(key))
                    add_branch(tree=branch, data=value)
                else:
                    tree.add(f'{key}: {value}')
        add_branch(tree=rt, data=self)
        return rt

@dataclasses.dataclass(slots=True, frozen=False)
class Range:
    """
    Wrapper around :ncu_report:`IRange`.

    This class loads the actions that are in the range.

    If both :py:attr:`includes` and :py:attr:`excludes` are empty, load all actions in :py:attr:`range`.
    """
    index: int
    range: typing.Any = dataclasses.field(init=False)
    actions: tuple[Action, ...] = dataclasses.field(init=False)

    report: dataclasses.InitVar[typing.Any]
    includes: dataclasses.InitVar[typing.Iterable[str] | None] = None
    excludes: dataclasses.InitVar[typing.Iterable[str] | None] = None

    def __post_init__(self, report, includes: typing.Iterable[str] | None, excludes: typing.Iterable[str] | None) -> None:
        self.range = report.range_by_idx(self.index)

        if not includes and not excludes:
            self.actions = tuple(Action(nvtx_range=self.range, index=iaction) for iaction in range(self.range.num_actions()))
        else:
            self.actions = tuple(Action(nvtx_range=self.range, index=iaction) for iaction in self.range.actions_by_nvtx(includes or [], excludes or []))

    def __repr__(self) -> str:
        return f'{self.range} (index {self.index})'

@dataclasses.dataclass(slots=True, frozen=False)
class Action:
    """
    Wrapper around :ncu_report:`IAction`.
    """
    index: int
    action: typing.Any = dataclasses.field(init=False)
    domains: tuple[NvtxDomain, ...] = dataclasses.field(init=False, default=())

    nvtx_range: dataclasses.InitVar[typing.Any]

    def __post_init__(self, nvtx_range) -> None:
        self.action = nvtx_range.action_by_idx(self.index)

        if (nvtx_state := self.action.nvtx_state()) is not None:
            self.domains = tuple(NvtxDomain(nvtx_state=nvtx_state, index=idomain) for idomain in nvtx_state.domains())

    def __repr__(self) -> str:
        return f"{self.action} (index {self.index}, domains {self.domains}, mangled {self.action.name(self.action.NameBase_MANGLED)})"

@dataclasses.dataclass(slots=True, frozen=False)
class NvtxDomain:
    """
    Wrapper around :ncu_report:`INvtxDomainInfo`.
    """
    index: int
    nvtx_domain: typing.Any = dataclasses.field(init=False)

    nvtx_state: dataclasses.InitVar[typing.Any]

    def __post_init__(self, nvtx_state) -> None:
        self.nvtx_domain = nvtx_state.domain_by_id(self.index)

    def __repr__(self) -> str:
        return f"{self.nvtx_domain} (index {self.index}, name {self.nvtx_domain.name()} {self.nvtx_domain.start_end_ranges()} {self.nvtx_domain.push_pop_ranges()})"

def load_ncu_report() -> types.ModuleType:
    """
    Attempt to load the `Nsight Compute` `Python` interface (``ncu_report``).

    Priority:

    #. If ``ncu_report`` is already imported, return it.
    #. Try a regular ``import ncu_report`` (for users who set ``PYTHONPATH``).
    #. Try to locate ``ncu`` via ``shutil.which``, deduce version, and load
       the bundled `Python` module from `Nsight Compute` installation path.
    """
    if 'ncu_report' in sys.modules:
        return sys.modules['ncu_report']

    try:
        importlib.import_module('ncu_report')
        return sys.modules['ncu_report']
    except ImportError:
        pass

    ncu = shutil.which('ncu')
    if ncu is None:
        raise RuntimeError("'ncu' was not found.")

    version_string = subprocess.check_output((ncu, '--version')).decode().splitlines()[-1]
    if (matched := re.match(r'Version ([0-9.]+)', version_string)) is not None:
        version_full = matched.groups()[0]
    else:
        raise RuntimeError(f'unexpected format {version_string}')
    if (matched := re.match(r'[0-9]+.[0-9]+.[0-9]+', version_full)) is not None:
        version_short = matched.group()
        logging.debug(f'Found \'ncu\' ({ncu}) with version {version_full}.')
    else:
        raise RuntimeError(f'unexpected format {version_string}')

    nsight_compute_dir = pathlib.Path('/opt') / 'nvidia' / 'nsight-compute' / version_short
    if not nsight_compute_dir.is_dir():
        raise FileExistsError(f'Presumed Nsight Compute install directory {nsight_compute_dir} does not exist.')

    nsight_compute_extra_python_dir = nsight_compute_dir / 'extras' / 'python'
    ncu_report_spec = importlib.util.spec_from_file_location(
        name="ncu_report",
        location=nsight_compute_extra_python_dir / 'ncu_report.py',
        submodule_search_locations=[str(nsight_compute_extra_python_dir)],
    )
    assert ncu_report_spec is not None
    ncu_report = importlib.util.module_from_spec(ncu_report_spec)
    sys.modules['ncu_report'] = ncu_report
    assert ncu_report_spec.loader is not None
    ncu_report_spec.loader.exec_module(ncu_report)
    return sys.modules['ncu_report']

class Report:
    """
    This class is a wrapper around the `Python` tool provided by NVIDIA `Nsight Compute`
    to parse its reports.

    In particular, the NVIDIA Python tool (``ncu_report``) provides low-level access to
    the collected data by iterating over ranges and actions. This class uses these functionalities
    to extract all the collected data into a custom data structure of type :py:class:`ProfilingResults`.
    This data structures is a nested tree data structure that provides a higher level, direct access
    to the data of interest by NVTX range (if NVTX is used) and by demangled kernel name.

    References:

    * https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#python-report-interface
    * https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    def __init__(self, *, path: pathlib.Path | None = None, name: str | None = None, command: Command | None = None) -> None:
        """
        Load the report ``<path>/<name>.ncu-rep`` or the report generated by :py:class:`reprospect.tools.ncu.session.Command`.
        """
        self.ncu_report = load_ncu_report()

        # Load the report.
        if path is not None and name is not None and command is None:
            self.path = path / (name + '.ncu-rep')
            if not self.path.is_file():
                raise FileNotFoundError(f"The report path {self.path} is not a file.")
        elif path is None and name is None and command is not None:
            self.path = command.output.with_suffix('.ncu-rep')
        else:
            raise RuntimeError

        self.report = self.ncu_report.load_report(self.path)

        logging.info(f"Report {self.path} loaded successfully ({self.report}).")

    def extract_results_in_range(
        self,
        metrics: typing.Collection[MetricKind],
        range_idx: int = 0,
        includes: typing.Iterable[str] | None = None,
        excludes: typing.Iterable[str] | None = None,
        demangler: type[CuppFilt | LlvmCppFilt] | None = None,
    ) -> ProfilingResults:
        """
        Extract the `metrics` of the actions in the range with ID `range_idx`.
        Possibly filter by NVTX with `includes` and `excludes`.

        :param metrics: Must be iterable from start to end many times.
        """
        logging.info(f'Collecting metrics {metrics} in report {self.path} for range {range_idx} with include filters {includes} and exclude filters {excludes}.')

        current_range = Range(report=self.report, index=range_idx, includes=includes, excludes=excludes)

        logging.debug(f'Parsing range {current_range}.')

        profiling_results = ProfilingResults()

        if not current_range.actions:
            raise RuntimeError('no action found')

        # Loop over the actions.
        for action in current_range.actions:

            logging.debug(f'Parsing action {action.action.name()} ({action}).')

            results = self.collect_metrics_from_action(action=action, metrics=metrics)

            results['mangled']   = action.action.name(action.action.NameBase_MANGLED)
            results['demangled'] = action.action.name(action.action.NameBase_DEMANGLED) if demangler is None else demangler.demangle(typing.cast(str, results['mangled']))

            # Loop over the domains of the action.
            # Note that domains are only available if NVTX was enabled during collection.
            if action.domains:
                for domain in action.domains:
                    profiling_results.assign_metrics(
                        accessors=domain.nvtx_domain.push_pop_ranges() + (f'{action.action.name()}-{action.index}',),
                        data=ProfilingMetrics(results),
                    )
            else:
                profiling_results.assign_metrics((f'{action.action.name()}-{action.index}',), ProfilingMetrics(results))

        return profiling_results

    def collect_metrics_from_action(self, *, metrics: typing.Iterable[MetricKind], action: Action) -> dict[str, ProfilingMetricData]:
        """
        Collect values of the `metrics` in the `action`.

        References:

        * https://github.com/shunting314/gpumisc/blob/37bbb827ae2ed6f5777daff06956c7a10aafe34d/ncu-related/official-sections/FPInstructions.py#L51
        * https://github.com/NVIDIA/nsight-training/blob/2d680f7f8368b945bc00b22834808af24eff4c3d/cuda/nsight_compute/python_report_interface/Opcode_instanced_metrics.ipynb
        """
        results: dict[str, ProfilingMetricData] = {}

        for metric in metrics:
            if isinstance(metric, MetricCorrelation):
                metric_correlation = action.action.metric_by_name(metric.name)

                correlated: MetricCorrelationDataType = {}
                correlations = metric_correlation.correlation_ids()
                assert correlations.num_instances() == metric_correlation.num_instances()
                for icor in range(metric_correlation.num_instances()):
                    key = self.get_metric_value(metric=correlations, index=icor)
                    assert isinstance(key, int | str)
                    correlated[key] = metric_correlation.value(icor)
                value = self.get_metric_value(metric=metric_correlation)
                assert isinstance(value, ValueType)
                results[metric.name] = MetricCorrelationData(
                    value=value,
                    correlated=correlated,
                )
            elif isinstance(metric, Metric):
                assert metric.pretty_name is not None
                metric_data = self.fill_metric(action=action, metric=metric)
                if metric.subs is not None:
                    assert isinstance(metric_data, dict)
                    for sub, value in metric_data.items():
                        results[f'{metric.pretty_name}.{sub}'] = value
                else:
                    assert metric_data is not None
                    results[metric.pretty_name] = metric_data
            elif isinstance(metric, MetricDeviceAttribute):
                results[metric.full_name] = self.get_metric_by_name(action=action, metric=metric.full_name).value()
            else:
                raise NotImplementedError(metric)

        return results

    def get_metric_value(self, metric: typing.Any, index: int | None = None) -> ValueType | str:
        """
        Recent ``ncu`` (>= 2025.3.0.0) provide a `value` method.
        """
        if hasattr(self.ncu_report, 'IMetric_kind_to_value_func'):
            converter = self.ncu_report.IMetric_kind_to_value_func[metric.kind()]
            return converter(metric, index) if index is not None else converter(metric)
        return metric.value(idx=index)

    @classmethod
    def fill_metric(cls, action: Action, metric: Metric) -> MetricData:
        """
        Loop over submetrics of `metric`.
        """
        if metric.subs is not None:
            return {sub: cls.get_metric_by_name(action=action, metric=f'{metric.name}.{sub}').value() for sub in metric.subs}
        return cls.get_metric_by_name(action=action, metric=metric.name).value()

    @classmethod
    def get_metric_by_name(cls, *, action: Action, metric: str):
        """
        Read a `metric` in `action`.
        """
        collected = action.action.metric_by_name(metric)
        if collected is None:
            raise RuntimeError(f"There was a problem retrieving metric '{metric}'. It probably does not exist.")
        return collected
