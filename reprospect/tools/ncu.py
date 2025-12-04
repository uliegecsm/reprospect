# pylint: disable=too-many-lines

import collections.abc
import dataclasses
import enum
import functools
import importlib
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
import typing

import attrs
import blake3
import rich.tree

from reprospect.tools import cacher
from reprospect.tools.binaries import CuppFilt, LlvmCppFilt
from reprospect.utils import ldd
from reprospect.utils import rich_helpers

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Unit(StrEnum):
    """
    Available units.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-decoder
    """
    L1TEX = 'l1tex'
    SM    = 'sm'
    SMSP  = 'smsp'

class PipeStage(StrEnum):
    """
    Available pipe stages.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder
    """
    TAG = 't'
    TAG_OUTPUT = 't_output'

class Quantity(StrEnum):
    """
    Available quantities.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder
    """
    INSTRUCTION = 'inst'
    REQUEST     = 'requests'
    SECTOR      = 'sectors'
    WAVEFRONT   = 'wavefronts'

@dataclasses.dataclass(frozen = True, slots = True)
class DeviceAttributeMetric:
    """
    ``ncu`` device attribute metric, such as::

        device__attribute_architecture

    .. note::

        Available device attribute metrics can be queryied with::

            ncu --query-metrics-collection=device
    """
    name : str

    @property
    def full_name(self) -> str:
        return f'device__attribute_{self.name}'

    def gather(self) -> tuple[str]:
        return (self.full_name,)

#: A single metric value type.
ValueType : typing.TypeAlias = int | float

#: A single metric value type or a dictionary of submetric values of such type.
MetricData : typing.TypeAlias = ValueType | dict[str, ValueType]

@dataclasses.dataclass(frozen = False, slots = True)
class Metric:
    """
    Used to represent a ``ncu`` metric.

    If :py:attr:`subs` is not given, it is assumed that :py:attr:`name` is a valid metric
    that can be directly evaluated by ``ncu``.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    #: The base name of the metric.
    name : str

    #: Human readable name.
    pretty_name : str | None = None

    #: Optional sub-metric names.
    subs : tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.pretty_name = self.pretty_name or self.name

    def gather(self) -> tuple[str, ...]:
        """
        Get the list of sub-metric names or the metric name itself if no sub-metrics are defined.
        """
        if self.subs is not None:
            return tuple(f'{self.name}.{sub}' for sub in self.subs)
        return (self.name,)

class MetricCounterRollUp(StrEnum):
    """
    Available roll-ups for :py:class:`MetricCounter`.
    """
    SUM = enum.auto()
    AVG = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()

@dataclasses.dataclass(slots = True, frozen = False)
class MetricCounter(Metric):
    """
    A counter metric.

    The sub-metric names are expected to be from :py:class:`MetricCounterRollUp`.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """

class MetricRatioRollUp(StrEnum):
    """
    Available roll-ups for :py:class:`MetricRatio`.
    """
    PCT = enum.auto()
    RATIO = enum.auto()
    MAX_RATE = enum.auto()

@dataclasses.dataclass(slots = True, frozen = False)
class MetricRatio(Metric):
    """
    A ratio metric.

    The sub-metric names are expected to be from :py:class:`MetricRatioRollUp`.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """

class XYZBase:
    """
    Base class for factories used to represent a triplet of metrics in `x`, `y` and `z` dimensions.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-reference
    """
    prefix : typing.ClassVar[str]

    @classmethod
    def create(cls, dims : typing.Iterable[str] | None = None) -> typing.Iterable[Metric]:
        if not dims:
            dims = ('x', 'y', 'z')
        return (Metric(name = cls.prefix + dim) for dim in dims)

class LaunchBlock(XYZBase):
    """
    Factory of metrics ``launch__block_dim_x``, ``launch__block_dim_y`` and ``launch__block_dim_z``.
    """
    prefix : typing.ClassVar[str] = 'launch__block_dim_'

class LaunchGrid(XYZBase):
    """
    Factory of metrics ``launch__grid_dim_x``, ``launch__grid_dim_y`` and ``launch__grid_dim_z``.
    """
    prefix : typing.ClassVar[str] = 'launch__grid_dim_'

MetricCorrelationDataType : typing.TypeAlias = dict[str | int, ValueType]

@dataclasses.dataclass(frozen = True, slots = True)
class MetricCorrelationData:
    """
    Data for :py:class:`MetricCorrelation`.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    correlated : MetricCorrelationDataType
    value : ValueType | None = None

@dataclasses.dataclass(frozen = True, slots = True)
class MetricCorrelation:
    """
    A metric with correlations, like ``sass__inst_executed_per_opcode``.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    name : str

    def gather(self) -> tuple[str]:
        return (self.name,)

def counter_name_from(
    *,
    unit : Unit,
    pipestage : typing.Optional[PipeStage] = None,
    quantity : Quantity | str,
    qualifier : typing.Optional[str] = None
) -> str:
    """
    Based on ``ncu`` metrics naming convention:

        ``unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)``

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    name = f'{unit}__'
    if pipestage:
        name += f'{pipestage}_'
    name += f'{quantity}'
    if qualifier:
        name += f'_{qualifier}'
    return name

class L1TEXCacheGlobalLoadInstructions:
    """
    Factory of counter metric ``(unit)__(sass?)_inst_executed_op_global_ld``.
    """
    @staticmethod
    def create(*,
        unit : Unit = Unit.SMSP,
        mode : typing.Literal['sass'] | None = 'sass',
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        name = counter_name_from(
            unit = unit,
            quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
            qualifier = 'executed_op_global_ld',
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, 'instructions', mode or ''))

        return MetricCounter(name = name, pretty_name = pretty_name, subs = subs)

class L1TEXCacheGlobalLoadRequests:
    """
    Factory of counter metric ``l1tex__t_requests_pipe_lsu_mem_global_op_ld``.
    """
    @staticmethod
    def create(*,
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        name = counter_name_from(
            unit = Unit.L1TEX,
            pipestage = PipeStage.TAG,
            quantity = Quantity.REQUEST,
            qualifier = 'pipe_lsu_mem_global_op_ld',
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, 'requests'))

        return MetricCounter(name = name, pretty_name = pretty_name, subs = subs)

class L1TEXCacheGlobalLoadSectors:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld``.
    """
    @staticmethod
    def create(*,
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
        suffix : typing.Optional[typing.Literal['hit', 'miss']] = None,
    ) -> 'MetricCounter':
        qualifier = f'pipe_lsu_mem_global_op_ld_lookup_{suffix}' if suffix else 'pipe_lsu_mem_global_op_ld'

        name = counter_name_from(
            unit = Unit.L1TEX,
            pipestage = PipeStage.TAG,
            quantity = Quantity.SECTOR,
            qualifier = qualifier,
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, f'sectors {suffix}' if suffix else 'sectors'))

        return MetricCounter(name = name, pretty_name = pretty_name, subs = subs)

class L1TEXCacheGlobalLoadSectorHits:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit``.
    """
    @staticmethod
    def create(*,
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        return L1TEXCacheGlobalLoadSectors.create(subs = subs, suffix = 'hit')

class L1TEXCacheGlobalLoadSectorMisses:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss``.
    """
    @staticmethod
    def create(*,
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        return L1TEXCacheGlobalLoadSectors.create(subs = subs, suffix = 'miss')

class L1TEXCacheGlobalLoadWavefronts:
    """
    Factory of counter metric ``l1tex__t_wavefronts_pipe_lsu_mem_global_op_ld``.
    """
    @staticmethod
    def create(*,
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        name = counter_name_from(
            unit = Unit.L1TEX,
            pipestage = PipeStage.TAG_OUTPUT,
            quantity = Quantity.WAVEFRONT,
            qualifier = 'pipe_lsu_mem_global_op_ld',
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, 'wavefronts'))

        return MetricCounter(name = name, pretty_name = pretty_name, subs = subs)

class L1TEXCacheGlobalLoad:

    NAME : typing.Final[str] = 'global load'

    Instructions : typing.Final[typing.Type[L1TEXCacheGlobalLoadInstructions]] = L1TEXCacheGlobalLoadInstructions # pylint: disable=invalid-name

    Requests : typing.Final[typing.Type[L1TEXCacheGlobalLoadRequests]] = L1TEXCacheGlobalLoadRequests # pylint: disable=invalid-name

    Sectors : typing.Final[typing.Type[L1TEXCacheGlobalLoadSectors]] = L1TEXCacheGlobalLoadSectors # pylint: disable=invalid-name

    SectorHits : typing.Final[typing.Type[L1TEXCacheGlobalLoadSectorHits]] = L1TEXCacheGlobalLoadSectorHits # pylint: disable=invalid-name

    SectorMisses : typing.Final[typing.Type[L1TEXCacheGlobalLoadSectorMisses]] = L1TEXCacheGlobalLoadSectorMisses # pylint: disable=invalid-name

    Wavefronts : typing.Final[typing.Type[L1TEXCacheGlobalLoadWavefronts]] = L1TEXCacheGlobalLoadWavefronts # pylint: disable=invalid-name

class L1TEXCacheGlobalStoreInstructions:
    """
    Factory of counter metric ``(unit)__(sass?)_inst_executed_op_global_st``.
    """
    @staticmethod
    def create(*,
        unit : Unit = Unit.SMSP,
        mode : typing.Literal['sass'] | None = 'sass',
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        name = counter_name_from(
            unit = unit,
            quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
            qualifier = 'executed_op_global_st',
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalStore.NAME, 'instructions', mode or ''))

        return MetricCounter(name = name, pretty_name = pretty_name, subs = subs)

class L1TEXCacheGlobalStoreSectors:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_st``.
    """
    @staticmethod
    def create(*,
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        name = counter_name_from(
            unit = Unit.L1TEX,
            pipestage = PipeStage.TAG,
            quantity = Quantity.SECTOR,
            qualifier = 'pipe_lsu_mem_global_op_st',
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalStore.NAME, 'sectors'))

        return MetricCounter(name = name, pretty_name = pretty_name, subs = subs)

class L1TEXCacheGlobalStore:

    NAME : typing.Final[str] = 'global store'

    Instructions : typing.Final[typing.Type[L1TEXCacheGlobalStoreInstructions]] = L1TEXCacheGlobalStoreInstructions # pylint: disable=invalid-name

    Sectors : typing.Final[typing.Type[L1TEXCacheGlobalStoreSectors]] = L1TEXCacheGlobalStoreSectors # pylint: disable=invalid-name

class L1TEXCacheLocalStoreInstructions:
    """
    Factory of counter metric ``(unit)__(sass?)_inst_executed_op_local_st``.
    """
    @staticmethod
    def create(*,
        unit : Unit = Unit.SMSP,
        mode : typing.Literal['sass'] | None = 'sass',
        subs : tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> 'MetricCounter':
        name = counter_name_from(
            unit = unit,
            quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
            qualifier = 'executed_op_local_st',
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.LocalStore.NAME, 'instructions', mode or ''))

        return MetricCounter(name = name, pretty_name = pretty_name, subs = subs)

class L1TEXCacheLocalStore:

    NAME : typing.Final[str] = 'local store'

    Instructions : typing.Final[typing.Type[L1TEXCacheLocalStoreInstructions]] = L1TEXCacheLocalStoreInstructions # pylint: disable=invalid-name

class L1TEXCache:
    """
    A selection of metrics related to `L1/TEX` cache.

    See :cite:`nvidia-ncu-requests-wavefronts-sectors`.
    """
    NAME : typing.Final[str] = 'L1/TEX cache'

    GlobalLoad : typing.Final[typing.Type[L1TEXCacheGlobalLoad]] = L1TEXCacheGlobalLoad # pylint: disable=invalid-name

    GlobalStore : typing.Final[typing.Type[L1TEXCacheGlobalStore]] = L1TEXCacheGlobalStore # pylint: disable=invalid-name

    LocalStore : typing.Final[typing.Type[L1TEXCacheLocalStore]] = L1TEXCacheLocalStore # pylint: disable=invalid-name

MetricKind : typing.TypeAlias = Metric | MetricCorrelation | DeviceAttributeMetric

def gather(metrics : typing.Iterable[MetricKind]) -> tuple[str, ...]:
    """
    Retrieve all sub-metric names, e.g. to pass them to ``ncu``.
    """
    return tuple(name for metric in metrics for name in metric.gather())

@attrs.define(frozen = True, slots = True)
class SessionCommand:
    """
    Used by :py:class:`reprospect.tools.ncu.Session`.
    """
    opts: list[str]                             #: ``ncu`` options that do not involve paths.
    output: pathlib.Path                        #: ``ncu`` report file.
    log: pathlib.Path                           #: ``ncu`` log file.
    metrics: typing.Iterable[MetricKind] | None #: ``ncu`` metrics.
    executable: str | pathlib.Path              #: Executable to run.
    args: list[str | pathlib.Path] | None       #: Arguments to pass to the executable.

    @functools.cached_property
    def to_tuple(self) -> tuple[str | pathlib.Path, ...]:
        """
        Build the full ``ncu`` command.
        """
        cmd : list[str | pathlib.Path] = ['ncu']

        if self.opts:
            cmd += self.opts

        cmd += ['--force-overwrite', '-o', self.output]
        cmd += ['--log-file', self.log]

        if self.metrics:
            cmd.append(f'--metrics={",".join(gather(metrics = self.metrics))}')

        cmd.append(self.executable)

        if self.args:
            cmd += self.args

        return tuple(cmd)

class Session:
    """
    `Nsight Compute` session interface.
    """
    def __init__(self, *, output : pathlib.Path):
        self.output = output

    def get_command(self, *,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_includes : typing.Optional[typing.Iterable[str]] = None,
        metrics : typing.Optional[typing.Iterable[MetricKind]] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
    ) -> SessionCommand:
        """
        Create a :py:class:`SessionCommand`.
        """
        if not opts:
            opts = []

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

        return SessionCommand(
            opts = opts,
            output = self.output,
            log = self.output.with_suffix('.log'),
            metrics = metrics,
            executable = executable,
            args = args,
        )

    def run(
        self,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_includes : typing.Optional[typing.Iterable[str]] = None,
        metrics : typing.Optional[typing.Iterable[MetricKind]] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
        cwd : typing.Optional[pathlib.Path] = None,
        env : typing.Optional[typing.MutableMapping] = None,
        retries : typing.Optional[int] = None,
        sleep : typing.Callable[[int, int], float] = lambda retry, retries: 3. * (1. - retry / retries),
    ) -> SessionCommand:
        """
        Run ``ncu``.

        :param nvtx_includes: Refer to https://docs.nvidia.com/nsight-compute/2023.3/NsightComputeCli/index.html#nvtx-filtering.
        :param retries: ``ncu`` might fail acquiring some resources because other instances are running. Retry a few times.
                        See https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq (Profiling failed because a driver resource was unavailable).

        :param sleep: The time to sleep between successive retries.
                      The callable is given the current retry index (descending) and the amount of allowed retries.

        .. warning::

            According to https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#ElevPrivsTag,
            `GPU` performance counters are not available to all users by default.

        .. note::

            As of ``ncu`` `2025.1.1.0`, a note tells us that specified NVTX include expressions match only start/end ranges.

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

        if retries is None:
            retries = 1

        for retry in reversed(range(retries)):
            try:
                logging.info(f"Launching 'ncu' with {command.to_tuple} (log file at {command.log}).")
                subprocess.check_call(command.to_tuple, cwd = cwd, env = env)
                break
            except subprocess.CalledProcessError:
                retry_allowed = False
                if retry > 0 and command.log.is_file():
                    with open(self.output.with_suffix('.log'), mode = 'r', encoding = 'utf-8') as fin:
                        for line in fin:
                            if line.startswith('==ERROR== Profiling failed because a driver resource was unavailable.'):
                                logging.warning('Retrying because a driver resource was unavailable.')
                                retry_allowed = True
                                break

                if not retry_allowed:
                    logging.exception(
                        f"Failed launching 'ncu' with {command}."
                        "\n"
                        f"{command.log.read_text(encoding = 'utf-8') if command.log.is_file() else ''}"
                    )
                    raise
                sleep_for = sleep(retry, retries)
                logging.info(f'Sleeping {sleep_for} seconds before retrying.')
                time.sleep(sleep_for)

        return command

@dataclasses.dataclass(slots = True, frozen = False)
class Range:
    """
    Wrapper around :ncu_report:`IRange`.

    This class loads the actions that are in the range.

    If both :py:attr:`includes` and :py:attr:`excludes` are empty, load all actions in :py:attr:`range`.
    """
    index : int
    range : typing.Any = dataclasses.field(init = False)
    actions : tuple['Action', ...] = dataclasses.field(init = False)

    report : dataclasses.InitVar[typing.Any]
    includes : dataclasses.InitVar[typing.Iterable[str] | None] = None
    excludes : dataclasses.InitVar[typing.Iterable[str] | None] = None

    def __post_init__(self, report, includes : typing.Iterable[str] | None = None, excludes : typing.Iterable[str] | None = None) -> None:
        self.range = report.range_by_idx(self.index)

        if not includes and not excludes:
            self.actions = tuple(Action(nvtx_range = self.range, index = iaction) for iaction in range(self.range.num_actions()))
        else:
            self.actions = tuple(Action(nvtx_range = self.range, index = iaction) for iaction in self.range.actions_by_nvtx(includes if includes else [], excludes if excludes else []))

    def __repr__(self) -> str:
        return f'{self.range} (index {self.index})'

@dataclasses.dataclass(slots = True, frozen = False)
class Action:
    """
    Wrapper around :ncu_report:`IAction`.
    """
    index : int
    action : typing.Any = dataclasses.field(init = False)
    domains : tuple['NvtxDomain', ...] = dataclasses.field(init = False, default = ())

    nvtx_range : dataclasses.InitVar[typing.Any]

    def __post_init__(self, nvtx_range) -> None:
        self.action = nvtx_range.action_by_idx(self.index)

        if (nvtx_state := self.action.nvtx_state()) is not None:
            self.domains = tuple(NvtxDomain(nvtx_state = nvtx_state, index = idomain) for idomain in nvtx_state.domains())

    def __repr__(self) -> str:
        return f"{self.action} (index {self.index}, domains {self.domains}, mangled {self.action.name(self.action.NameBase_MANGLED)})"

@dataclasses.dataclass(slots = True, frozen = False)
class NvtxDomain:
    """
    Wrapper around :ncu_report:`INvtxDomainInfo`.
    """
    index : int
    nvtx_domain : typing.Any = dataclasses.field(init = False)

    nvtx_state : dataclasses.InitVar[typing.Any]

    def __post_init__(self, nvtx_state) -> None:
        self.nvtx_domain = nvtx_state.domain_by_id(self.index)

    def __repr__(self) -> str:
        return f"{self.nvtx_domain} (index {self.index}, name {self.nvtx_domain.name()} {self.nvtx_domain.start_end_ranges()} {self.nvtx_domain.push_pop_ranges()})"

#: Metric data in profiling results.
ProfilingMetricData : typing.TypeAlias = MetricData | MetricCorrelationData | str

class ProfilingMetrics(collections.abc.Mapping[str, ProfilingMetricData]):
    """
    Mapping of profiling metric keys to their values.

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('data',)

    def __init__(self, data : dict[str, ProfilingMetricData]) -> None:
        self.data : typing.Final[dict[str, ProfilingMetricData]] = data

    def __getitem__(self, key : str, /) -> ProfilingMetricData:
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

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    __slots__ = ('data',)

    def __init__(self, data : dict[str, 'ProfilingResults | ProfilingMetrics'] | None = None) -> None:
        self.data : typing.Final[dict[str, 'ProfilingResults | ProfilingMetrics']] = data if data is not None else {}

    def query(self, accessors : typing.Iterable[str]) -> typing.Union['ProfilingResults', ProfilingMetrics]:
        """
        Get the internal node in the hierarchy or the leaf node with profiling metrics
        at accessor path `accessors`.
        """
        current : ProfilingResults | ProfilingMetrics = self
        for accessor in accessors:
            if not isinstance(current, ProfilingResults):
                raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r} instead.')
            current = current.data[accessor]
        return current

    def query_metrics(self, accessors : typing.Iterable[str]) -> ProfilingMetrics:
        """
        Query the accessor path `accessors`, check that it leads to a leaf node with profiling metrics,
        and return this leaf node with profiling metrics.
        """
        current = self.query(accessors = accessors)
        if not isinstance(current, ProfilingMetrics):
            raise TypeError(f'Expecting leaf node with profiling metrics at {accessors}, got {type(current).__name__!r} instead.')
        return current

    def query_single_next(self, accessors : typing.Iterable[str]) -> tuple[str, typing.Union['ProfilingResults', ProfilingMetrics]]:
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
        current = self.query(accessors = accessors)
        if isinstance(current, ProfilingResults):
            if len(current) != 1:
                raise RuntimeError(f'Expecting a single entry at {accessors}, got {len(current)} entries instead.')
            return next(iter(current.data.items()))
        raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r} instead.')

    def query_single_next_metrics(self, accessors : typing.Iterable[str]) -> tuple[str, ProfilingMetrics]:
        """
        Query the accessor path `accessors`, check that it leads to an internal node with exactly
        one entry, check that this entry is a leaf node with profiling metrics, and return this
        leaf node with profiling metrics.
        """
        key, value = self.query_single_next(accessors = accessors)
        if not isinstance(value, ProfilingMetrics):
            raise TypeError(f'Expecting leaf node {key!r} with profiling metrics as the single entry at {accessors}, got {type(value).__name__!r} instead.')
        return key, value

    def iter_metrics(self, accessors : typing.Iterable[str] = ()) -> typing.Generator[tuple[str, ProfilingMetrics], None, None]:
        """
        Query the accessor path `accessors`, check that it leads to an internal node, check that all entries
        are leaf nodes with profiling metrics, and return an iterator over these leaf nodes with profiling metrics.
        """
        current = self.query(accessors = accessors)
        if not isinstance(current, ProfilingResults):
            raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r} instead.')
        for key, value in current.data.items():
            if not isinstance(value, ProfilingMetrics):
                raise TypeError(f'Expecting entry {key!r} to be a leaf node with profiling metrics at {accessors}.')
            yield key, value

    def assign_metrics(self, accessors: typing.Sequence[str], data : ProfilingMetrics) -> None:
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

    def aggregate_metrics(self, accessors : typing.Iterable[str], keys : typing.Optional[typing.Iterable[str]] = None) -> dict[str, int | float]:
        """
        Aggregate metric values across multiple leaf nodes with profiling metrics at accessor path `accessors`.

        :param keys: Specific metric keys to aggregate. If :py:obj:`None`, uses all keys from the first leaf node.
        """
        current = self.query(accessors = accessors)

        if not isinstance(current, ProfilingResults):
            raise TypeError(f'Expecting internal node at {accessors}, got {type(current).__name__!r}.')

        if keys is None:
            value = next(iter(current.data.values()))
            assert isinstance(value, ProfilingMetrics)
            keys = value.keys()

        return {
            key : sum(
                v for _, m in current.iter_metrics(accessors = ())
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
        def add_branch(*, tree : rich.tree.Tree, data : ProfilingResults | ProfilingMetrics) -> None:
            for key, value in data.data.items() if isinstance(data, ProfilingResults) else data.items():
                if isinstance(value, ProfilingResults | ProfilingMetrics):
                    branch = tree.add(str(key))
                    add_branch(tree = branch, data = value)
                else:
                    tree.add(f'{key}: {value}')
        add_branch(tree = rt, data = self)
        return rt

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
        name                       = "ncu_report",
        location                   = nsight_compute_extra_python_dir / 'ncu_report.py',
        submodule_search_locations = [str(nsight_compute_extra_python_dir)],
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
    def __init__(self, *, path : typing.Optional[pathlib.Path] = None, name : typing.Optional[str] = None, session : typing.Optional[Session] = None) -> None:
        """
        Load the report ``<path>/<name>.ncu-rep`` or the report generated by :py:class:`Session`.
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

    def extract_results_in_range(
        self,
        range_idx : int,
        metrics : typing.Collection[MetricKind],
        includes : typing.Optional[typing.Iterable[str]] = None,
        excludes : typing.Optional[typing.Iterable[str]] = None,
        demangler : typing.Optional[typing.Type[CuppFilt | LlvmCppFilt]] = None,
    ) -> ProfilingResults:
        """
        Extract the `metrics` of the actions in the range with ID `range_idx`.
        Possibly filter by NVTX with `includes` and `excludes`.

        :param metrics: Must be iterable from start to end many times.
        """
        logging.info(f'Collecting metrics {metrics} in report {self.path} for range {range_idx} with include filters {includes} and exclude filters {excludes}.')

        current_range = Range(report = self.report, index = range_idx, includes = includes, excludes = excludes)

        logging.debug(f'Parsing range {current_range}.')

        profiling_results = ProfilingResults()

        if not current_range.actions:
            raise RuntimeError('no action found')

        # Loop over the actions.
        for action in current_range.actions:

            logging.debug(f'Parsing action {action.action.name()} ({action}).')

            results = self.collect_metrics_from_action(action = action, metrics = metrics)

            results['mangled']   = action.action.name(action.action.NameBase_MANGLED)
            results['demangled'] = action.action.name(action.action.NameBase_DEMANGLED) if demangler is None else demangler.demangle(typing.cast(str, results['mangled']))

            # Loop over the domains of the action.
            # Note that domains are only available if NVTX was enabled during collection.
            if action.domains:
                for domain in action.domains:
                    profiling_results.assign_metrics(
                        accessors = domain.nvtx_domain.push_pop_ranges() + (f'{action.action.name()}-{action.index}',),
                        data = ProfilingMetrics(results),
                    )
            else:
                profiling_results.assign_metrics((f'{action.action.name()}-{action.index}',), ProfilingMetrics(results))

        return profiling_results

    def collect_metrics_from_action(self, *, metrics : typing.Iterable[MetricKind], action : Action) -> dict[str, ProfilingMetricData]:
        """
        Collect values of the `metrics` in the `action`.

        References:

        * https://github.com/shunting314/gpumisc/blob/37bbb827ae2ed6f5777daff06956c7a10aafe34d/ncu-related/official-sections/FPInstructions.py#L51
        * https://github.com/NVIDIA/nsight-training/blob/2d680f7f8368b945bc00b22834808af24eff4c3d/cuda/nsight_compute/python_report_interface/Opcode_instanced_metrics.ipynb
        """
        results : dict[str, ProfilingMetricData] = {}

        for metric in metrics:
            if isinstance(metric, MetricCorrelation):
                metric_correlation = action.action.metric_by_name(metric.name)

                correlated : MetricCorrelationDataType = {}
                correlations = metric_correlation.correlation_ids()
                assert correlations.num_instances() == metric_correlation.num_instances()
                for icor in range(metric_correlation.num_instances()):
                    key = self.get_metric_value(metric = correlations, index = icor)
                    assert isinstance(key, int | str)
                    correlated[key] = metric_correlation.value(icor)
                value = self.get_metric_value(metric = metric_correlation)
                assert isinstance(value, ValueType)
                results[metric.name] = MetricCorrelationData(
                    value = value,
                    correlated = correlated,
                )
            elif isinstance(metric, Metric):
                assert metric.pretty_name is not None
                metric_data = self.fill_metric(action = action, metric = metric)
                if metric.subs is not None:
                    assert isinstance(metric_data, dict)
                    for sub, value in metric_data.items():
                        results[f'{metric.pretty_name}.{sub}'] = value
                else:
                    assert metric_data is not None
                    results[metric.pretty_name] = metric_data
            elif isinstance(metric, DeviceAttributeMetric):
                results[metric.full_name] = self.get_metric_by_name(action = action, metric = metric.full_name).value()
            else:
                raise NotImplementedError(metric)

        return results

    def get_metric_value(self, metric : typing.Any, index : int | None = None) -> ValueType | str:
        """
        Recent ``ncu`` (>= 2025.3.0.0) provide a `value` method.
        """
        if hasattr(self.ncu_report, 'IMetric_kind_to_value_func'):
            converter = self.ncu_report.IMetric_kind_to_value_func[metric.kind()]
            return converter(metric, index) if index is not None else converter(metric)
        return metric.value(idx = index)

    @classmethod
    def fill_metric(cls, action : Action, metric : Metric) -> MetricData:
        """
        Loop over submetrics of `metric`.
        """
        if metric.subs is not None:
            return {sub : cls.get_metric_by_name(action = action, metric = f'{metric.name}.{sub}').value() for sub in metric.subs}
        return cls.get_metric_by_name(action = action, metric = metric.name).value()

    @classmethod
    def get_metric_by_name(cls, *, action : Action, metric : str):
        """
        Read a `metric` in `action`.
        """
        collected = action.action.metric_by_name(metric)
        if collected is None:
            raise RuntimeError(f"There was a problem retrieving metric '{metric}'. It probably does not exist.")
        return collected

class Cacher(cacher.Cacher):
    """
    Cacher tailored to ``ncu`` results.

    ``ncu`` require quite some time to acquire results, especially when there are many kernels to profile and/or
    many metrics to collect.

    On a cache hit, the cacher will serve:

    - ``<cache_key>.ncu-rep`` file
    - ``.log`` file

    On a cache miss, ``ncu`` is launched and the cache entry populated accordingly.

    .. note::

        It is assumed that hashing is faster than running ``ncu`` itself.

    .. warning::

        The cache should not be shared between machines, since there may be differences between machines
        that influence the results but are not included in the hashing.
    """
    TABLE : typing.ClassVar[str] = 'ncu'

    def __init__(self, *, session : Session, directory : typing.Optional[str | pathlib.Path] = None):
        super().__init__(directory = pathlib.Path(directory or os.environ['HOME']) / '.ncu-cache')
        self.session = session

    def hash_impl(self, *,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_includes : typing.Optional[typing.Iterable[str]] = None,
        metrics : typing.Optional[typing.Iterable[MetricKind]] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
        env : typing.Optional[typing.MutableMapping] = None,
    ) -> blake3.blake3:
        """
        Hash based on:

        * ``ncu`` version
        * ``ncu`` options (but not the output and log files)
        * ``ncu`` metrics
        * executable content
        * executable arguments
        * linked libraries
        * environment
        """
        hasher = blake3.blake3() # pylint: disable=not-callable

        hasher.update(subprocess.check_output(('ncu', '--version')))

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
            hasher.update(shlex.join(map(str, command.args)).encode())

        for lib in sorted(ldd.get_shared_dependencies(file = command.executable)):
            hasher.update_mmap(lib)

        if env:
            hasher.update(json.dumps(env).encode())

        return hasher

    @override
    def hash(self, **kwargs) -> blake3.blake3:
        return self.hash_impl(
            executable = kwargs['executable'],
            opts = kwargs.get('opts'),
            nvtx_includes = kwargs.get('nvtx_includes'),
            metrics = kwargs.get('metrics'),
            args = kwargs.get('args'),
            env = kwargs.get('env'),
        )

    @override
    def populate(self, directory : pathlib.Path, **kwargs) -> SessionCommand:
        """
        When there is a cache miss, call :py:meth:`reprospect.tools.ncu.Session.run`.
        Fill the `directory` with the artifacts.
        """
        command = self.session.run(**kwargs)

        shutil.copy(dst = directory, src = command.output.with_suffix('.ncu-rep'))
        shutil.copy(dst = directory, src = command.log)

        return command

    def run(self, **kwargs) -> cacher.Cacher.Entry:
        """
        On a cache hit, copy files from the cache entry.
        """
        entry = self.get(**kwargs)

        if entry.cached:
            shutil.copytree(entry.directory, self.session.output.parent, dirs_exist_ok = True)

        return entry
