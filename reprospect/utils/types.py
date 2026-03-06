"""
Type information for numeric scalar types.

The minimal information needed to describe a numeric type is:

* **kind** and **signedness** — integer or floating-point, signed or unsigned (see :py:class:`~reprospect.utils.types.Kind`)
* **bits** — width in bits (*e.g.* 16, 32, 64, 128)

This module provides :py:class:`~reprospect.utils.types.TypeInfo` to carry this information.

.. note::

    Several alternatives to writing this module have been considered:

    * :py:mod:`numpy` does not support arbitrarily wide types such as ``int128``.
    * https://github.com/jax-ml/ml_dtypes focuses on small types used in machine learning.
"""

from __future__ import annotations

import dataclasses
import enum
import sys
import typing

import numpy
import numpy.typing

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

class Kind(enum.Enum):
    INT = enum.auto()
    UINT = enum.auto()
    FLOAT = enum.auto()

    @property
    def signed(self) -> bool:
        return self in {Kind.INT, Kind.FLOAT}

    @classmethod
    def from_numpy(cls, kind: str) -> Self:
        """
        See https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html.
        """
        match kind:
            case 'i':
                return cls.INT
            case 'u':
                return cls.UINT
            case 'f':
                return cls.FLOAT
            case _:
                raise ValueError(kind)

@dataclasses.dataclass(frozen=True, slots=True)
class TypeInfo:
    """
    Minimal type information.
    """
    bits: int
    """
    Number of bits.
    """

    kind: Kind

    @property
    def signed(self) -> bool:
        return self.kind.signed

    @property
    def itemsize(self) -> int:
        return self.bits // 8

    @classmethod
    def normalize(cls, *, dtype: ConvertibleTypeInfo) -> Self:
        if isinstance(dtype, TypeInfo):
            return typing.cast(Self, dtype)
        if isinstance(dtype, int):
            return cls(bits=dtype, kind=Kind.INT)
        if isinstance(dtype, numpy.dtype):
            return cls(bits=dtype.itemsize * 8, kind=Kind.from_numpy(kind=dtype.kind))
        if isinstance(dtype, type) and issubclass(dtype, numpy.generic):
            tmp = numpy.dtype(dtype)
            return cls(bits=tmp.itemsize * 8, kind=Kind.from_numpy(kind=tmp.kind))
        raise TypeError(f"Cannot convert {dtype!r} to {cls!r}.")

    def __repr__(self):
        return f'{self.kind.name.lower()}{self.bits}'

ConvertibleTypeInfo: typing.TypeAlias = TypeInfo | int | numpy.typing.DTypeLike
