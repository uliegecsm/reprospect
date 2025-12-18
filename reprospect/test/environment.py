import dataclasses
import os
import typing

T = typing.TypeVar('T')
"""Type variable for :py:class:`~EnvironmentField` and related generics."""

@dataclasses.dataclass(slots=True)
class EnvironmentField(typing.Generic[T]):
    """
    Descriptor that returns a value lazily read from an environment variable.

    Based on:

    * https://docs.python.org/3/howto/descriptor.html
    * https://mypy.readthedocs.io/en/stable/generics.html#defining-generic-classes
    """
    env: str | None = None
    """
    Name of the environment variable.
    """

    converter: typing.Callable[[str], T] | None = None
    """
    Callable to convert the value of the environment variable to the target type.
    """

    default: T | None = None
    """
    Default value if the environment variable does not exist.
    """

    _cached: T | None = dataclasses.field(default=None, init=False, repr=False)
    """
    Value, cached.
    """

    _attr_name: str | None = dataclasses.field(default=None, init=False, repr=False)
    """
    Name of the attribute.
    """

    def __post_init__(self) -> None:
        if self.default is not None and self.converter is None:
            self.converter = type(self.default)

    def __set_name__(self, owner: type, name: str) -> None:
        """
        References:

        * https://docs.python.org/3/reference/datamodel.html#object.__set_name__
        """
        self._attr_name = name

    @typing.overload
    def __get__(self, instance: None, owner: type) -> "EnvironmentField[T]": ...

    @typing.overload
    def __get__(self, instance: object, owner: type) -> T: ...

    def __get__(self, instance, owner=None):
        """
        References:

        * https://docs.python.org/3/reference/datamodel.html#object.__get__
        """
        if instance is None:
            return self

        return self.read(instance=instance, owner=owner)

    def read(self, instance, owner) -> T:
        """
        Read from the environment.
        """
        if self._cached is not None:
            return self._cached

        key = self.env or self._attr_name

        if key is None:
            raise AttributeError("Descriptor not initialized properly.")

        if (value := os.getenv(key)) is not None and self.converter is not None:
            self._cached = self.converter(value)
        elif self.default is not None:
            self._cached = self.default
        else:
            raise RuntimeError(f"Missing required environment variable {key!r} or converter or default value for {instance.__class__ if instance else owner}.")

        return self._cached

    def reset(self) -> None:
        """
        Reset, thus forcing a re-read from environment on next access.
        """
        self._cached = None
