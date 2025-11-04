import copy
import dataclasses
import functools
import importlib
import json
import logging
import operator
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

@dataclasses.dataclass(frozen = False)
class Metric:
    """
    Used to represent a ``ncu`` metric.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    #: The base name of the metric.
    name : str

    #: Human readable name.
    human : str

    #: A dictionary of sub-metrics and their value.
    subs : dict[typing.Any, int | float | None]

    def __init__(self,
        name : str,
        *,
        subs : typing.Optional[typing.Iterable[typing.Any]] = None,
        human : typing.Optional[str] = None,
    ) -> None:
        """
        If `subs` is not given, it is assumed that `name` is a valid metric
        that can be directly evaluated by ``ncu``.
        """
        self.name  = name
        self.human = human if human else self.name
        self.subs  = {x: None for x in subs} if subs else {'' : None}

    def __repr__(self) -> str:
        return self.name + '(' + ','.join([f'{sub}={value}' for sub, value in self.subs.items()]) + ')'

    def gather(self) -> list[str]:
        """
        Get the list of sub-metric names or the metric name itself if no sub-metrics are defined.
        """
        return [f'{self.name}{sub}' for sub in self.subs.keys()]

class MetricCounterRollUp(StrEnum):
    """
    Available roll-ups for :py:class:`MetricCounter`.
    """
    SUM = '.sum'
    AVG = '.avg'
    MIN = '.min'
    MAX = '.max'

class MetricCounter(Metric):
    """
    A counter metric.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    RollUp : typing.Final[typing.Type[MetricCounterRollUp]] = MetricCounterRollUp # pylint: disable=invalid-name

class MetricRatioRollUp(StrEnum):
    """
    Available roll-ups for :py:class:`MetricRatio`.
    """
    PCT      = '.pct'
    RATIO    = '.ratio'
    MAX_RATE = '.max_rate'

class MetricRatio(Metric):
    """
    A ratio metric.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    RollUp : typing.Final[typing.Type[MetricRatioRollUp]] = MetricRatioRollUp # pylint: disable=invalid-name

class XYZBase:
    """
    Base class for factories used to represent a triplet of metrics in `x`, `y` and `z` dimensions.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-reference
    """
    prefix : typing.ClassVar[str]

    @classmethod
    def get(cls, dims : typing.Optional[typing.Iterable[str]] = None) -> typing.Iterable[Metric]:
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

@dataclasses.dataclass(frozen = False)
class MetricCorrelation:
    """
    A metric with correlations, like ``sass__inst_executed_per_opcode``.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    name : str
    correlated : dict[str, int] | None = None
    value : float | None = None

    def gather(self) -> list[str]:
        return [self.name]

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

class L1TEXCacheGlobalLoadInstructions(MetricCounter):
    """
    Counter metric ``(unit)__(sass?)_inst_executed_op_global_ld``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None, unit : Unit = Unit.SMSP, mode : typing.Optional[typing.Literal['sass']] = 'sass') -> None:
        super().__init__(
            name = counter_name_from(
                unit = unit,
                quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
                qualifier = 'executed_op_global_ld',
            ), subs = subs or (MetricCounterRollUp.SUM,),
            human = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, 'instructions', mode or '')),
        )

class L1TEXCacheGlobalLoadRequests(MetricCounter):
    """
    Counter metric ``l1tex__t_requests_pipe_lsu_mem_global_op_ld``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None) -> None:
        super().__init__(
            name = counter_name_from(
                unit = Unit.L1TEX,
                pipestage = PipeStage.TAG,
                quantity = Quantity.REQUEST,
                qualifier = 'pipe_lsu_mem_global_op_ld',
            ), subs = subs or (MetricCounterRollUp.SUM),
            human = ' '.join([L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, 'requests']),
        )

class L1TEXCacheGlobalLoadSectors(MetricCounter):
    """
    Counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld``.
    """
    def __init__(self,
        subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None,
        suffix : typing.Optional[typing.Literal['hit', 'miss']] = None,
    ) -> None:
        qualifier = f'pipe_lsu_mem_global_op_ld_lookup_{suffix}' if suffix else 'pipe_lsu_mem_global_op_ld'

        super().__init__(
            name = counter_name_from(
                unit = Unit.L1TEX,
                pipestage = PipeStage.TAG,
                quantity = Quantity.SECTOR,
                qualifier = qualifier,
            ),
            subs = subs or (MetricCounterRollUp.SUM,),
            human = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, f'sectors {suffix}' if suffix else 'sectors')),
        )

class L1TEXCacheGlobalLoadSectorHits(L1TEXCacheGlobalLoadSectors):
    """
    Counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None) -> None:
        L1TEXCacheGlobalLoadSectors.__init__(self, subs = subs, suffix = 'hit')

class L1TEXCacheGlobalLoadSectorMisses(L1TEXCacheGlobalLoadSectors):
    """
    Counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None) -> None:
        super().__init__(subs = subs, suffix = 'miss')

class L1TEXCacheGlobalLoadWavefronts(MetricCounter):
    """
    Counter metric ``l1tex__t_wavefronts_pipe_lsu_mem_global_op_ld``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None) -> None:
        super().__init__(
            name = counter_name_from(
                unit = Unit.L1TEX,
                pipestage = PipeStage.TAG_OUTPUT,
                quantity = Quantity.WAVEFRONT,
                qualifier = 'pipe_lsu_mem_global_op_ld',
            ), subs = subs or (MetricCounterRollUp.SUM,),
            human = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, 'wavefronts')),
        )

class L1TEXCacheGlobalLoad:

    NAME : typing.Final[str] = 'global load'

    Instructions : typing.Final[typing.Type[L1TEXCacheGlobalLoadInstructions]] = L1TEXCacheGlobalLoadInstructions # pylint: disable=invalid-name

    Requests : typing.Final[typing.Type[L1TEXCacheGlobalLoadRequests]] = L1TEXCacheGlobalLoadRequests # pylint: disable=invalid-name

    Sectors : typing.Final[typing.Type[L1TEXCacheGlobalLoadSectors]] = L1TEXCacheGlobalLoadSectors # pylint: disable=invalid-name

    SectorHits : typing.Final[typing.Type[L1TEXCacheGlobalLoadSectorHits]] = L1TEXCacheGlobalLoadSectorHits # pylint: disable=invalid-name

    SectorMisses : typing.Final[typing.Type[L1TEXCacheGlobalLoadSectorMisses]] = L1TEXCacheGlobalLoadSectorMisses # pylint: disable=invalid-name

    Wavefronts : typing.Final[typing.Type[L1TEXCacheGlobalLoadWavefronts]] = L1TEXCacheGlobalLoadWavefronts # pylint: disable=invalid-name

class L1TEXCacheGlobalStoreInstructions(MetricCounter):
    """
    Counter metric ``(unit)__(sass?)_inst_executed_op_global_st``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None, unit : Unit = Unit.SMSP, mode : typing.Optional[typing.Literal['sass']] = 'sass') -> None:
        super().__init__(
            name = counter_name_from(
                unit = unit,
                quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
                qualifier = 'executed_op_global_st',
            ), subs = subs or (MetricCounterRollUp.SUM,),
            human = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalStore.NAME, 'instructions', mode or '')),
        )

class L1TEXCacheGlobalStoreSectors(MetricCounter):
    """
    Counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_st``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None) -> None:
        super().__init__(
            name = counter_name_from(
                unit = Unit.L1TEX,
                pipestage = PipeStage.TAG,
                quantity = Quantity.SECTOR,
                qualifier = 'pipe_lsu_mem_global_op_st',
            ), subs = subs or (MetricCounterRollUp.SUM,),
            human = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalStore.NAME, 'sectors')),
        )

class L1TEXCacheGlobalStore:

    NAME : typing.Final[str] = 'global store'

    Instructions : typing.Final[typing.Type[L1TEXCacheGlobalStoreInstructions]] = L1TEXCacheGlobalStoreInstructions # pylint: disable=invalid-name

    Sectors : typing.Final[typing.Type[L1TEXCacheGlobalStoreSectors]] = L1TEXCacheGlobalStoreSectors # pylint: disable=invalid-name

class L1TEXCacheLocalStoreInstructions(MetricCounter):
    """
    Counter metric ``(unit)__(sass?)_inst_executed_op_local_st``.
    """
    def __init__(self, subs : typing.Optional[typing.Iterable[MetricCounterRollUp]] = None, unit : Unit = Unit.SMSP, mode : typing.Optional[typing.Literal['sass']] = 'sass') -> None:
        super().__init__(
            name = counter_name_from(
                unit = unit,
                quantity = f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
                qualifier = 'executed_op_local_st',
            ), subs = subs or (MetricCounterRollUp.SUM,),
            human = ' '.join((L1TEXCache.NAME, L1TEXCache.LocalStore.NAME, 'instructions', mode or '')),
        )

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

def gather(metrics : typing.Iterable[Metric | MetricCorrelation]) -> list[str]:
    """
    Retrieve all sub-metric names, e.g. to pass them to ``ncu``.
    """
    return [name for metric in metrics for name in metric.gather()]

@attrs.define(frozen = True, slots = True)
class SessionCommand:
    """
    Used by :py:class:`reprospect.tools.ncu.Session`.
    """
    opts: list[str]                                      #: ``ncu`` options that do not involve paths.
    output: pathlib.Path                                 #: ``ncu`` report file.
    log: pathlib.Path                                    #: ``ncu`` log file.
    metrics: typing.Iterable[Metric | MetricCorrelation] | None #: ``ncu`` metrics.
    executable: str | pathlib.Path                       #: Executable to run.
    args: list[str | pathlib.Path] | None                #: Arguments to pass to the executable.

    @functools.cached_property
    def to_list(self) -> list[str | pathlib.Path]:
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

        return cmd

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
        metrics : typing.Optional[typing.Iterable[Metric | MetricCorrelation]] = None,
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
        metrics : typing.Optional[typing.Iterable[Metric | MetricCorrelation]] = None,
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
                logging.info(f"Launching 'ncu' with {command.to_list} (log file at {command.log}).")
                subprocess.check_call(command.to_list, cwd = cwd, env = env)
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

class Range:
    """
    Wrapper around :ncu_report:`IRange`.

    This class loads the actions that are in the range.
    """
    def __init__(self, report, index : int, includes : typing.Optional[list[str]] = None, excludes : typing.Optional[list[str]] = None) -> None:
        self.index = index
        self.range = report.range_by_idx(self.index)

        self.actions : list[Action] = []

        self._load_actions_by_nvtx(includes = includes, excludes = excludes)

    def _load_actions_by_nvtx(self, includes : typing.Optional[list[str]] = None, excludes : typing.Optional[list[str]] = None) -> None:
        """
        If both `includes` and `excludes` are empty, load all actions.
        """
        if not includes and not excludes:
            for iaction in range(self.range.num_actions()):
                self.actions.append(Action(nvtx_range = self.range, index = iaction))
        else:
            for iaction in self.range.actions_by_nvtx(includes if includes else [], excludes if excludes else []):
                self.actions.append(Action(nvtx_range = self.range, index = iaction))

    def __repr__(self) -> str:
        return f'{self.range} (index {self.index})'

class Action:
    """
    Wrapper around :ncu_report:`IAction`.
    """
    def __init__(self, nvtx_range, index : int) -> None:
        self.index  = index
        self.action = nvtx_range.action_by_idx(index)

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

#: A single metric value in profiling results.
MetricValue : typing.TypeAlias = typing.Union[MetricCorrelation, int, float, str, None]

#: Profiling results for a single entry.
ProfilingResult : typing.TypeAlias = dict[str, MetricValue]

#: Nested profiling results data structure.
NestedProfilingResults : typing.TypeAlias = dict[str, typing.Union['NestedProfilingResults', MetricValue]]

class ProfilingResults(rich_helpers.TreeMixin, NestedProfilingResults):
    """
    Data structure for storing profiling results.

    This data structure is a nested (ordered) dictionary.
    It has helper functions that ease dealing with nested values.
    """
    def query(self, accessors : typing.Iterable[str]) -> typing.Union[NestedProfilingResults, ProfilingResult, MetricValue]:
        """
        Get (nested) value from the `accessors` path.
        """
        return functools.reduce(operator.getitem, accessors, typing.cast(dict, self))

    def set(self, accessors: typing.Sequence[str], data : dict[str, MetricValue]) -> None:
        """
        Set or create (nested) `data` from the `accessors` path.
        Creates intermediate dictionaries as needed.
        """
        current = typing.cast(dict, self)
        for accessor in accessors[0:-1]:
            current = current.setdefault(accessor, {})
        current[accessors[-1]] = data

    def aggregate(self, accessors : typing.Iterable[str], keys : typing.Optional[typing.Iterable] = None) -> dict:
        """
        Aggregate values of dictionaries selected by `accessors`.

        The selected dictionaries are assumed to be at the last nesting level and to all have the same keys.
        """
        # Get all dictionaries that match 'accessors'.
        results = self.query(accessors = accessors)

        if not isinstance(results, dict):
            raise TypeError(f'Expected dictionary at {accessors}, got {type(results).__name__}.')

        samples = typing.cast(dict[str, dict[str, MetricValue]], results)

        # Create the aggregate results.
        # Get keys of the first sample if no keys provided. We assume all samples have the same keys.
        if keys is None:
            keys = samples[next(iter(samples.keys()))].keys()

        return {key: sum(s[key] for s in samples.values()) for key in keys}

    @override
    def to_tree(self) -> rich.tree.Tree:
        def add_branch(*, tree : rich.tree.Tree, data : dict) -> None:
            for key, value in data.items():
                if isinstance(value, dict):
                    branch = tree.add(key)
                    add_branch(tree = branch, data = value)
                else:
                    tree.add(f'{key}: {value}')

        tree = rich.tree.Tree('Profiling results')
        add_branch(tree = tree, data = self)

        return tree

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

    version_string = subprocess.check_output([ncu, '--version']).decode().splitlines()[-1]
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
    This data structures is a nested dictionary that provides a more direct access to the data
    of interest by NVTX range (if NVTX is used) and by demangled kernel name.

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

    def extract_metrics_in_range(
        self,
        range_idx : int,
        metrics : typing.Iterable[Metric | MetricCorrelation],
        includes : typing.Optional[list[str]] = None,
        excludes : typing.Optional[list[str]] = None,
        demangler : typing.Optional[typing.Type[CuppFilt | LlvmCppFilt]] = None,
    ) -> ProfilingResults:
        """
        Extract the `metrics` of the actions in the range with ID `range_idx`.
        Possibly filter by NVTX with `includes` and `excludes`.
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
            results['demangled'] = action.name(action.NameBase_DEMANGLED) if demangler is None else demangler.demangle(typing.cast(str, results['mangled']))

            # Loop over the domains of the action.
            # Note that domains are only available if NVTX was enabled during collection.
            if action.domains:
                for domain in action.domains:
                    profiling_results.set(
                        accessors = domain.push_pop_ranges() + (f'{action.name()}-{action.index}',),
                        data = results,
                    )
            else:
                profiling_results.set((f'{action.name()}-{action.index}',), results)

        return profiling_results

    def collect_metrics_from_action(self, *, metrics : typing.Iterable[Metric | MetricCorrelation], action : Action) -> dict[str, MetricValue]:
        """
        Collect values of the `metrics` in the `action`.

        References:

        * https://github.com/shunting314/gpumisc/blob/37bbb827ae2ed6f5777daff06956c7a10aafe34d/ncu-related/official-sections/FPInstructions.py#L51
        * https://github.com/NVIDIA/nsight-training/blob/2d680f7f8368b945bc00b22834808af24eff4c3d/cuda/nsight_compute/python_report_interface/Opcode_instanced_metrics.ipynb
        """
        results : dict[str, MetricValue] = {}

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
    def fill_metric(cls, action : Action, metric : Metric) -> Metric:
        """
        Loop over submetrics of `metric`.
        """
        for sub in metric.subs:
            metric.subs[sub] = cls.get_metric_by_name(action = action, metric = metric.name + sub).value()
        return metric

    @classmethod
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
        super().__init__(directory = directory if directory is not None else pathlib.Path(os.environ['HOME']) / '.ncu-cache')
        self.session = session

    def hash_impl(self, *,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_includes : typing.Optional[list[str]] = None,
        metrics : typing.Optional[typing.Iterable[Metric | MetricCorrelation]] = None,
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
