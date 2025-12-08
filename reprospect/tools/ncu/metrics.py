import dataclasses
import enum
import sys
import typing

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

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

@dataclasses.dataclass(frozen = True, slots = True)
class MetricDeviceAttribute:
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
    unit : Unit,
    pipestage : PipeStage | None = None,
    quantity : Quantity | str,
    qualifier : str | None = None
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
        suffix : typing.Literal['hit', 'miss'] | None = None,
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

MetricKind : typing.TypeAlias = Metric | MetricCorrelation | MetricDeviceAttribute

def gather(metrics : typing.Iterable[MetricKind]) -> tuple[str, ...]:
    """
    Retrieve all sub-metric names, e.g. to pass them to ``ncu``.
    """
    return tuple(name for metric in metrics for name in metric.gather())
