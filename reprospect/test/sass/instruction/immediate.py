import typing

from reprospect.test.sass.instruction.pattern import PatternBuilder


class Immediate:
    """
    Immediate patterns.
    """
    INF: typing.Final[str] = r'[+-]?INF'

    QNAN: typing.Final[str] = PatternBuilder.any(r'[+-]?QNAN', '0x7fffffff')
    """Quiet NaN. Note that in SASS, :code:`CUDART_NAN_F` from ``math_constants.h`` is represented as ``0x7fffffff``."""

    FLOATING: typing.Final[str] = r'(?:-?\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?'
    """
    References:

    * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L34
    """

    FLOATING_OR_LIMIT: typing.Final[str] = PatternBuilder.any(INF, QNAN, FLOATING)

    @classmethod
    def floating(cls) -> str:
        """
        :py:attr:`FLOATING` with `operands` group.
        """
        return PatternBuilder.group(cls.FLOATING, group='operands')

    @classmethod
    def floating_or_limit(cls) -> str:
        """
        :py:attr:`FLOATING_OR_LIMIT` with `operands` group.
        """
        return PatternBuilder.group(cls.FLOATING_OR_LIMIT, group='operands')
