import numpy

from reprospect.utils import types


class TestKind:
    """
    Tests for :code:`reprospect.utils.types.Kind`.
    """
    def test_from_numpy(self) -> None:
        assert types.Kind.from_numpy(kind='f') == types.Kind.FLOAT
        assert types.Kind.from_numpy(kind='i') == types.Kind.INT
        assert types.Kind.from_numpy(kind='u') == types.Kind.UINT

class TestTypeInfo:
    """
    Tests for :code:`reprospect.utils.types.TypeInfo`.
    """
    def test_itemsize(self) -> None:
        assert types.TypeInfo(bits=128, kind=types.Kind.INT).itemsize == 16

    def test_normalize_from_int(self) -> None:
        assert types.TypeInfo.normalize(dtype=64) == types.TypeInfo(bits=64, kind=types.Kind.INT)

    def test_normalize_from_numpy(self) -> None:
        assert types.TypeInfo.normalize(dtype=numpy.float64) == types.TypeInfo(bits=64, kind=types.Kind.FLOAT)
        assert types.TypeInfo.normalize(dtype=numpy.dtype('int32')) == types.TypeInfo(bits=32, kind=types.Kind.INT)

    def test_normalize_from_self(self) -> None:
        dtype = types.TypeInfo(bits=128, kind=types.Kind.FLOAT)
        converted = types.TypeInfo.normalize(dtype=dtype)
        assert dtype == converted
        assert id(dtype) == id(converted)
