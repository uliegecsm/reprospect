import pathlib
import typing

import pytest

from reprospect.test.environment import EnvironmentField


class TestEnvironmentField:
    """
    Tests for :py:class:`reprospect.test.environment.EnvironmentField`.
    """
    def test_no_attribute_name_or_env_key(self) -> None:
        """
        Raises if neither an attribute name nor an environment key is given.
        """
        with pytest.raises(AttributeError,match = r'Descriptor not initialized properly.'):
            EnvironmentField().read(None, None)

    def test_read_str_converter(self, monkeypatch) -> None:
        """
        Environment variable read as `str`.
        """
        class Config:
            test_var = EnvironmentField(converter = str)

        monkeypatch.setenv('test_var', 'hello')

        config = Config()

        assert config.test_var == 'hello'
        assert isinstance(config.test_var, str)

    def test_read_int_converter(self, monkeypatch) -> None:
        """
        Environment variable read as `int`.
        """
        class Config:
            count = EnvironmentField(converter = int)

        monkeypatch.setenv('count', '42')

        config = Config()

        assert config.count == 42
        assert isinstance(config.count, int)

    def test_not_in_environment_use_default(self) -> None:
        """
        The value is initialized to the given default value.
        """
        class DefaultValue:
            var: typing.ClassVar[EnvironmentField[float]] = EnvironmentField(default = 6.66)
            other = EnvironmentField(default = pathlib.Path('my-default-path'))

        default_value = DefaultValue()

        assert default_value.var == 6.66
        assert isinstance(default_value.var, float)

        assert default_value.other == pathlib.Path('my-default-path')
        assert isinstance(default_value.other, pathlib.Path)

    def test_not_in_environment_no_default(self) -> None:
        """
        The attribute cannot be initialized.
        """
        class Raises:
            var = EnvironmentField[float]()
            other = EnvironmentField[float]()

        with pytest.raises(RuntimeError, match = "Missing required environment variable 'var' or converter or default value for <class"):
            assert Raises().var

        with pytest.raises(RuntimeError, match = "Missing required environment variable 'other' or converter or default value for <class"):
            assert Raises().other

    def test_in_environment_converted_with_env_key(self, monkeypatch) -> None:
        """
        The value is correctly initialized from the environment (given a key), and converted.
        """
        class ReadAndConvertValue:
            var: EnvironmentField[pathlib.Path] = EnvironmentField(env = 'my_Weird_NAME', converter = pathlib.Path)
            other = EnvironmentField(env = 'MY_OTHER_weird', converter = pathlib.Path)

        monkeypatch.setenv('my_Weird_NAME', 'my-nice/path.rst')

        read_and_convert_value = ReadAndConvertValue()

        assert read_and_convert_value.var == pathlib.Path('my-nice/path.rst')
        assert isinstance(read_and_convert_value.var, pathlib.Path)

        monkeypatch.setenv('MY_OTHER_weird', 'ola/hi/bonjour')

        assert read_and_convert_value.other == pathlib.Path('ola/hi/bonjour')
        assert isinstance(read_and_convert_value.other, pathlib.Path)

    def test_in_environment_converted_no_env_key(self, monkeypatch) -> None:
        """
        The attribute is correctly initialized from the environment (no key given), and converted.
        """
        class WeirdType:
            def __init__(self, value: str) -> None:
                self.computed: int = hash(value)

        class ReadAndConvertValue:
            var = EnvironmentField(converter = WeirdType)

        monkeypatch.setenv('var', 'my-nice/path.rst')

        read_and_convert_value = ReadAndConvertValue()

        assert read_and_convert_value.var.computed == hash('my-nice/path.rst')
        assert isinstance(read_and_convert_value.var, WeirdType)

    def test_value_cached_at_class_level(self, monkeypatch) -> None:
        """
        The value is shared among all instances.
        """
        class Config:
            value = EnvironmentField(converter = str)

        monkeypatch.setenv('value', 'first')

        config_a = Config()

        assert config_a.value == 'first'

        monkeypatch.setenv('value', 'second')

        assert config_a.value == 'first'

        config_b = Config()

        assert config_b.value == 'first'

    def test_converter_from_default_type(self, monkeypatch) -> None:
        """
        If no converter was provided, infer it from the type of the default value.
        """
        class Config:
            var = EnvironmentField(default = float(42.666))

        config = Config()

        monkeypatch.setenv('var', '666.42')

        assert config.var == 666.42
        assert isinstance(config.var, float)

    def test_reset(self, monkeypatch) -> None:
        class Config:
            value = EnvironmentField(converter = float)

        config = Config()

        monkeypatch.setenv('value', '0.61')

        assert config.value == 0.61

        Config.value.reset()

        monkeypatch.setenv('value', '1.23')

        assert config.value == 1.23
        assert isinstance(config.value, float)
