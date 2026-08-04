from __future__ import annotations

import dataclasses
import enum
import sys
import typing

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum


#: A single metric value type.
ValueType: typing.TypeAlias = int | float

#: A single metric value type or a dictionary of submetric values of such type.
MetricData: typing.TypeAlias = ValueType | dict[str, ValueType]

class Metric:
    """
    Used to represent a ``ncu`` metric.

    If :py:attr:`subs` is not given, it is assumed that :py:attr:`name` is a valid metric
    that can be directly evaluated by ``ncu``.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure

    .. note::

        It is not decorated with :py:func:`dataclasses.dataclass` because of https://github.com/mypyc/mypyc/issues/1061.
    """
    #: Sub-metric components implied by the metric kind, omitted from labels.
    SILENT_SUBS: typing.ClassVar[frozenset[str]] = frozenset()

    __slots__ = ('name', 'pretty_name', 'subs')

    def __init__(self, name: str, pretty_name: str | None = None, subs: tuple[str | tuple[str, ...], ...] | None = None) -> None:
        """
        :param name: The base name of the metric.
        :param pretty_name: Human readable name; defaults to :py:attr:`name`.
        :param subs: Sub-metric names. The constructor normalizes each to a path of components.
        """
        self.name: typing.Final[str] = name
        self.pretty_name: typing.Final[str | None] = pretty_name
        self.subs: typing.Final[tuple[tuple[str, ...], ...] | None] = tuple((sub,) if isinstance(sub, str) else sub for sub in subs) if subs is not None else None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Metric):
            return self.name == other.name and self.pretty_name == other.pretty_name and self.subs == other.subs
        return NotImplemented

    def gather(self) -> tuple[str, ...]:
        """
        Get the list of sub-metric names or the metric name itself if no sub-metrics are defined.
        """
        if self.subs is not None:
            return tuple('.'.join((self.name, *sub)) for sub in self.subs)
        return (self.name,)

    def labels(self) -> tuple[str, ...]:
        """
        Get the list of sub-metric labels. Parallel to :py:meth:`gather`, but uses the pretty name.
        """
        if self.pretty_name is None:
            return self.gather()
        if self.subs is None:
            return (self.pretty_name,)
        result = []
        for sub in self.subs:
            sub_labels = tuple(getattr(c, 'label', c) for c in sub if c not in self.SILENT_SUBS)
            result.append(f'{self.pretty_name} ({", ".join(sub_labels)})' if sub_labels else self.pretty_name)
        assert len(set(result)) == len(result), result
        return tuple(result)

class MetricCounterRollUpQuantity(StrEnum):
    """
    Available quantities for :py:class:`MetricCounterRollUp`.
    """
    @property
    def label(self) -> str:
        return {self.PCT_OF_PEAK_SUSTAINED_ACTIVE: '% of peak active', self.PCT_OF_PEAK_SUSTAINED_ELAPSED: '% of peak elapsed'}[self]

    PCT_OF_PEAK_SUSTAINED_ACTIVE = enum.auto()
    PCT_OF_PEAK_SUSTAINED_ELAPSED = enum.auto()

class MetricCounterRollUp(StrEnum):
    """
    Available roll-ups for :py:class:`MetricCounter`.
    """
    @property
    def label(self) -> str:
        return self.value

    SUM = enum.auto()
    AVG = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()

class MetricCounter(Metric):
    """
    A counter metric.

    The sub-metric names are expected to be from :py:class:`MetricCounterRollUp`.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    SILENT_SUBS: typing.ClassVar[frozenset[str]] = frozenset({MetricCounterRollUp.SUM})

class MetricRatioRollUp(StrEnum):
    """
    Available roll-ups for :py:class:`MetricRatio`.
    """
    @property
    def label(self) -> str:
        return {self.PCT: '%', self.RATIO: 'ratio', self.MAX_RATE: 'max rate'}[self]

    PCT = enum.auto()
    RATIO = enum.auto()
    MAX_RATE = enum.auto()

class MetricRatio(Metric):
    """
    A ratio metric.

    The sub-metric names are expected to be from :py:class:`MetricRatioRollUp`.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    SILENT_SUBS: typing.ClassVar[frozenset[str]] = frozenset({MetricRatioRollUp.RATIO})

@dataclasses.dataclass(frozen=True, slots=True)
class MetricDeviceAttribute:
    """
    ``ncu`` device attribute metric, such as::

        device__attribute_architecture

    .. note::

        Available device attribute metrics can be queryied with::

            ncu --query-metrics-collection=device
    """
    name: str

    @property
    def full_name(self) -> str:
        return f'device__attribute_{self.name}'

    def gather(self) -> tuple[str]:
        return (self.full_name,)

    def labels(self) -> tuple[str]:
        return self.gather()

MetricCorrelationDataType: typing.TypeAlias = dict[str | int, ValueType]

@dataclasses.dataclass(frozen=True, slots=True)
class MetricCorrelationData:
    """
    Data for :py:class:`MetricCorrelation`.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    correlated: MetricCorrelationDataType
    value: ValueType | None = None

@dataclasses.dataclass(frozen=True, slots=True)
class MetricCorrelation:
    """
    A metric with correlations, like ``sass__inst_executed_per_opcode``.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
    """
    name: str

    def gather(self) -> tuple[str]:
        return (self.name,)

    def labels(self) -> tuple[str]:
        return self.gather()

class XYZBase:
    """
    Base class for factories used to represent a triplet of metrics in `x`, `y` and `z` dimensions.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-reference
    """
    PREFIX: typing.ClassVar[str]

    PRETTY_PREFIX: typing.ClassVar[str | None] = None

    @classmethod
    def create(cls, dims: typing.Iterable[str] | None = None) -> tuple[Metric, ...]:
        if not dims:
            dims = ('x', 'y', 'z')
        if cls.PRETTY_PREFIX:
            return tuple(Metric(name=cls.PREFIX + dim, pretty_name=f'{cls.PRETTY_PREFIX} {dim}') for dim in dims)
        return tuple(Metric(name=cls.PREFIX + dim) for dim in dims)

class LaunchBlock(XYZBase):
    """
    Factory of metrics ``launch__block_dim_x``, ``launch__block_dim_y`` and ``launch__block_dim_z``.
    """
    PREFIX: typing.ClassVar[str] = 'launch__block_dim_'

    PRETTY_PREFIX: typing.ClassVar[str | None] = 'Launch block size'

class LaunchGrid(XYZBase):
    """
    Factory of metrics ``launch__grid_dim_x``, ``launch__grid_dim_y`` and ``launch__grid_dim_z``.
    """
    PREFIX: typing.ClassVar[str] = 'launch__grid_dim_'

    PRETTY_PREFIX: typing.ClassVar[str | None] = 'Launch grid size'

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

def counter_name_from(
    *,
    unit: Unit,
    pipestage: PipeStage | None = None,
    quantity: Quantity | str,
    qualifier: str | None = None,
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
        unit: Unit = Unit.SMSP,
        mode: typing.Literal['sass'] | None = 'sass',
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        name = counter_name_from(
            unit=unit,
            quantity=f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
            qualifier='executed_op_global_ld',
        )

        pretty_name = ' '.join(filter(None, (L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, mode, 'instructions', unit)))

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheGlobalLoadRequests:
    """
    Factory of counter metric ``l1tex__t_requests_pipe_lsu_mem_global_op_ld``.
    """
    @staticmethod
    def create(*,
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        name = counter_name_from(
            unit=Unit.L1TEX,
            pipestage=PipeStage.TAG,
            quantity=Quantity.REQUEST,
            qualifier='pipe_lsu_mem_global_op_ld',
        )

        pretty_name = f'{L1TEXCache.NAME} {L1TEXCache.GlobalLoad.NAME} requests'

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheGlobalLoadSectors:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld``.
    """
    @staticmethod
    def create(*,
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
        suffix: typing.Literal['hit', 'miss'] | None = None,
    ) -> tuple[MetricCounter, ...]:
        qualifier = f'pipe_lsu_mem_global_op_ld_lookup_{suffix}' if suffix else 'pipe_lsu_mem_global_op_ld'

        name = counter_name_from(
            unit=Unit.L1TEX,
            pipestage=PipeStage.TAG,
            quantity=Quantity.SECTOR,
            qualifier=qualifier,
        )

        pretty_name = ' '.join((L1TEXCache.NAME, L1TEXCache.GlobalLoad.NAME, f'sectors {suffix}' if suffix else 'sectors'))

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheGlobalLoadSectorHits:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit``.
    """
    @staticmethod
    def create(*,
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        return L1TEXCacheGlobalLoadSectors.create(subs=subs, suffix='hit')

class L1TEXCacheGlobalLoadSectorMisses:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss``.
    """
    @staticmethod
    def create(*,
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        return L1TEXCacheGlobalLoadSectors.create(subs=subs, suffix='miss')

class L1TEXCacheGlobalLoadWavefronts:
    """
    Factory of counter metric ``l1tex__t_wavefronts_pipe_lsu_mem_global_op_ld``.
    """
    @staticmethod
    def create(*,
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        name = counter_name_from(
            unit=Unit.L1TEX,
            pipestage=PipeStage.TAG_OUTPUT,
            quantity=Quantity.WAVEFRONT,
            qualifier='pipe_lsu_mem_global_op_ld',
        )

        pretty_name = f'{L1TEXCache.NAME} {L1TEXCache.GlobalLoad.NAME} wavefronts'

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheGlobalLoad:

    NAME: typing.Final[str] = 'global load'

    Instructions: typing.Final[type[L1TEXCacheGlobalLoadInstructions]] = L1TEXCacheGlobalLoadInstructions # pylint: disable=invalid-name

    Requests: typing.Final[type[L1TEXCacheGlobalLoadRequests]] = L1TEXCacheGlobalLoadRequests # pylint: disable=invalid-name

    Sectors: typing.Final[type[L1TEXCacheGlobalLoadSectors]] = L1TEXCacheGlobalLoadSectors # pylint: disable=invalid-name

    SectorHits: typing.Final[type[L1TEXCacheGlobalLoadSectorHits]] = L1TEXCacheGlobalLoadSectorHits # pylint: disable=invalid-name

    SectorMisses: typing.Final[type[L1TEXCacheGlobalLoadSectorMisses]] = L1TEXCacheGlobalLoadSectorMisses # pylint: disable=invalid-name

    Wavefronts: typing.Final[type[L1TEXCacheGlobalLoadWavefronts]] = L1TEXCacheGlobalLoadWavefronts # pylint: disable=invalid-name

class L1TEXCacheGlobalStoreInstructions:
    """
    Factory of counter metric ``(unit)__(sass?)_inst_executed_op_global_st``.
    """
    @staticmethod
    def create(*,
        unit: Unit = Unit.SMSP,
        mode: typing.Literal['sass'] | None = 'sass',
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        name = counter_name_from(
            unit=unit,
            quantity=f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
            qualifier='executed_op_global_st',
        )

        pretty_name = ' '.join(filter(None, (L1TEXCache.NAME, L1TEXCache.GlobalStore.NAME, mode, 'instructions', unit)))

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheGlobalStoreRequests:
    """
    Factory of counter metric ``l1tex__t_requests_pipe_lsu_mem_global_op_st``.
    """
    @staticmethod
    def create(*,
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        name = counter_name_from(
            unit=Unit.L1TEX,
            pipestage=PipeStage.TAG,
            quantity=Quantity.REQUEST,
            qualifier='pipe_lsu_mem_global_op_st',
        )

        pretty_name = f'{L1TEXCache.NAME} {L1TEXCache.GlobalStore.NAME} requests'

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheGlobalStoreSectors:
    """
    Factory of counter metric ``l1tex__t_sectors_pipe_lsu_mem_global_op_st``.
    """
    @staticmethod
    def create(*,
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        name = counter_name_from(
            unit=Unit.L1TEX,
            pipestage=PipeStage.TAG,
            quantity=Quantity.SECTOR,
            qualifier='pipe_lsu_mem_global_op_st',
        )

        pretty_name = f'{L1TEXCache.NAME} {L1TEXCache.GlobalStore.NAME} sectors'

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheGlobalStore:

    NAME: typing.Final[str] = 'global store'

    Instructions: typing.Final[type[L1TEXCacheGlobalStoreInstructions]] = L1TEXCacheGlobalStoreInstructions # pylint: disable=invalid-name

    Requests: typing.Final[type[L1TEXCacheGlobalStoreRequests]] = L1TEXCacheGlobalStoreRequests # pylint: disable=invalid-name

    Sectors: typing.Final[type[L1TEXCacheGlobalStoreSectors]] = L1TEXCacheGlobalStoreSectors # pylint: disable=invalid-name

class L1TEXCacheLocalStoreInstructions:
    """
    Factory of counter metric ``(unit)__(sass?)_inst_executed_op_local_st``.
    """
    @staticmethod
    def create(*,
        unit: Unit = Unit.SMSP,
        mode: typing.Literal['sass'] | None = 'sass',
        subs: tuple[MetricCounterRollUp, ...] = (MetricCounterRollUp.SUM,),
    ) -> tuple[MetricCounter, ...]:
        name = counter_name_from(
            unit=unit,
            quantity=f'sass_{Quantity.INSTRUCTION}' if mode == 'sass' else Quantity.INSTRUCTION,
            qualifier='executed_op_local_st',
        )

        pretty_name = ' '.join(filter(None, (L1TEXCache.NAME, L1TEXCache.LocalStore.NAME, mode, 'instructions', unit)))

        return (MetricCounter(name=name, pretty_name=pretty_name, subs=subs),)

class L1TEXCacheLocalStore:

    NAME: typing.Final[str] = 'local store'

    Instructions: typing.Final[type[L1TEXCacheLocalStoreInstructions]] = L1TEXCacheLocalStoreInstructions # pylint: disable=invalid-name

class L1TEXCache:
    """
    A selection of metrics related to `L1/TEX` cache.

    See :cite:`nvidia-ncu-requests-wavefronts-sectors`.
    """
    NAME: typing.Final[str] = 'L1/TEX cache'

    GlobalLoad: typing.Final[type[L1TEXCacheGlobalLoad]] = L1TEXCacheGlobalLoad # pylint: disable=invalid-name

    GlobalStore: typing.Final[type[L1TEXCacheGlobalStore]] = L1TEXCacheGlobalStore # pylint: disable=invalid-name

    LocalStore: typing.Final[type[L1TEXCacheLocalStore]] = L1TEXCacheLocalStore # pylint: disable=invalid-name

class WarpStallBase:
    """
    Base class for factories of warp-stall ratio metrics.
    """
    TEMPLATE_NAME: typing.Final[str] = 'smsp__average_warps_issue_stalled_{reason}_per_issue_active'
    TEMPLATE_LABEL: typing.Final[str] = 'Warp stall {reason}'

    #: The stall reason as it appears in the ``ncu`` metric name.
    REASON: typing.ClassVar[str]

    #: The stall reason as it appears in the label.
    PRETTY_REASON: typing.ClassVar[str]

    @classmethod
    def create(cls, *,
        subs: tuple[MetricRatioRollUp, ...] = (MetricRatioRollUp.RATIO,),
    ) -> tuple[MetricRatio, ...]:
        name = cls.TEMPLATE_NAME.format(reason=cls.REASON)

        pretty_name = cls.TEMPLATE_LABEL.format(reason=cls.PRETTY_REASON)

        return (MetricRatio(name=name, pretty_name=pretty_name, subs=subs),)

class WarpStallLGThrottle(WarpStallBase):
    """
    Factory of ratio metric ``smsp__average_warps_issue_stalled_lg_throttle_per_issue_active``.
    """
    REASON: typing.ClassVar[str] = 'lg_throttle'

    PRETTY_REASON: typing.ClassVar[str] = 'LG throttle'

class WarpStallLongScoreboard(WarpStallBase):
    """
    Factory of ratio metric ``smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active``.
    """
    REASON: typing.ClassVar[str] = 'long_scoreboard'

    PRETTY_REASON: typing.ClassVar[str] = 'Long scoreboard'

class WarpStallMIOThrottle(WarpStallBase):
    """
    Factory of ratio metric ``smsp__average_warps_issue_stalled_mio_throttle_per_issue_active``.
    """
    REASON: typing.ClassVar[str] = 'mio_throttle'

    PRETTY_REASON: typing.ClassVar[str] = 'MIO throttle'

class WarpStallShortScoreboard(WarpStallBase):
    """
    Factory of ratio metric ``smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active``.
    """
    REASON: typing.ClassVar[str] = 'short_scoreboard'

    PRETTY_REASON: typing.ClassVar[str] = 'Short scoreboard'

class WarpStall:
    """
    A selection of metrics related to warp stall reasons.

    References:

    * https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons
    """
    LGThrottle: typing.Final[type[WarpStallLGThrottle]] = WarpStallLGThrottle # pylint: disable=invalid-name

    LongScoreboard: typing.Final[type[WarpStallLongScoreboard]] = WarpStallLongScoreboard # pylint: disable=invalid-name

    MIOThrottle: typing.Final[type[WarpStallMIOThrottle]] = WarpStallMIOThrottle # pylint: disable=invalid-name

    ShortScoreboard: typing.Final[type[WarpStallShortScoreboard]] = WarpStallShortScoreboard # pylint: disable=invalid-name


MetricKind: typing.TypeAlias = Metric | MetricCorrelation | MetricDeviceAttribute

def gather(metrics: typing.Iterable[MetricKind]) -> tuple[str, ...]:
    """
    Retrieve all sub-metric names, e.g. to pass them to ``ncu``.

    Order follows `metrics`, then each metric's :py:attr:`~Metric.subs`; parallel to :py:func:`labels`.
    """
    return tuple(name for metric in metrics for name in metric.gather())

def labels(metrics: typing.Iterable[MetricKind]) -> tuple[str, ...]:
    """
    Retrieve all sub-metric labels, e.g. to lookup collected data by label.

    Order follows `metrics`, then each metric's :py:attr:`~Metric.subs`; parallel to :py:func:`gather`.
    """
    return tuple(label for metric in metrics for label in metric.labels())
