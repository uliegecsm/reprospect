import os

import pytest

from reprospect.test.environment import EnvironmentAwareMixin

class TestEnvironmentAwareMixin:
    """
    Tests for :py:class:`reprospect.test.environment.EnvironmentAwareMixin`.
    """
    class MyClass(EnvironmentAwareMixin):
        def __init__(self):
            self.my_var = 42

    @pytest.fixture(scope = 'function')
    @staticmethod
    def instance():
        return TestEnvironmentAwareMixin.MyClass()

    def test_is_attribute(self, instance):
        """
        The instance already has the attribute. It must not be fetched from the environment.
        """
        os.environ['my_var'] = '666'
        assert instance.my_var == 42

    def test_is_not_attribute(self, instance):
        """
        The instance does not have the attribute, so it is defined from the environment the first time it is requested.
        """
        os.environ['my_other_var'] = '42'
        assert instance.my_other_var == '42'

        os.environ['my_other_var'] = '666'
        assert instance.my_other_var == '42'

    def test_is_not_attribute_is_not_in_environment(self, instance):
        """
        The instance does not have the attribute, and it is not in the environment.
        """
        with pytest.raises(AttributeError, match = 'MyClass has no attribute \'this_will_RAISE\' and there is no environment variable \'this_will_RAISE\''):
            assert instance.this_will_RAISE == 666
