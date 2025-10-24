import os
import typing

import typeguard

class EnvironmentAwareMixin:
    """
    Base class that resolves missing instance attributes from the environment.

    .. note::
        A missing attribute is read the first time from the environment, and is then set as an attribute,
        such that an update of the environment won't affect the attribute value in subsequent calls.
    """
    #: Specify convertors from environment variable values to attributes of the relevant types.
    _REQUIRED_ATTRIBUTES_WITH_DEFAULT_FROM_ENV : typing.ClassVar[dict[str, typing.Callable[[str], typing.Any]]] = {}

    @typeguard.typechecked
    @classmethod
    def read_from_env(cls, name : str) -> typing.Any:
        value = os.getenv(name)
        if value is None:
            raise AttributeError(f'{cls.__name__} has no attribute {name!r} and there is no environment variable {name!r}.')

        if name in cls._REQUIRED_ATTRIBUTES_WITH_DEFAULT_FROM_ENV:
            value = cls._REQUIRED_ATTRIBUTES_WITH_DEFAULT_FROM_ENV[name](value)

        setattr(cls, name, value)

        return value

    @typeguard.typechecked
    def __getattr__(self, name : str) -> typing.Any:
        return self.read_from_env(name)
