import contextlib

import nvtx
import typeguard

@contextlib.contextmanager
@typeguard.typechecked
def push_pop_range(domain : nvtx._lib.lib.DummyDomain, **kwargs):
    domain.push_range(attributes = domain.get_event_attributes(**kwargs))
    try:
        yield
    finally:
        domain.pop_range()

@contextlib.contextmanager
@typeguard.typechecked
def start_end_range(domain : nvtx._lib.lib.DummyDomain, **kwargs):
    range_id = domain.start_range(attributes = domain.get_event_attributes(**kwargs))
    try:
        yield
    finally:
        domain.end_range(range_id)

class TestNVTX:
    """
    Use the `Python` interface of `NVTX` to create intricated situations.

    References:
        * https://nvidia.github.io/NVTX/python/
    """
    def function(self) -> None:
        pass

    def test_sanity(self):
        assert nvtx.enabled()

    def test_context_managed_annotation(self, request) -> None:
        """
        Context-managed annotation.
        """
        with nvtx.annotate(message = request.node.name, color = 'green'):
            self.function()

    def test_push_pop_range_in_domain(self, request) -> None:
        """
        Push/pop range in a domain.
        """
        with push_pop_range(domain = nvtx.Domain(name = request.node.name), message = 'osef'):
            self.function()

    def test_start_end_range_in_domain(self, request) -> None:
        """
        Start/end range in a domain.
        """
        with start_end_range(domain = nvtx.Domain(name = request.node.name), message = 'osef'):
            self.function()

    def test_registered_string(self, request) -> None:
        """
        Registered string.
        """
        domain = nvtx.Domain(request.node.name)

        reg = domain.get_registered_string(string = 'my very long string that is better made a registered string')

    def test_intricated(self, request) -> None:
        """
        Build a situation with many intricated ranges.
        """
        domain = nvtx.Domain(name = request.node.name)

        with start_end_range(domain = domain, message = 'start-end-level-0'):
            with push_pop_range(domain = domain, message = 'push-pop-level-1'):
                with push_pop_range(domain = domain, message = 'push-pop-level-2'):
                    with push_pop_range(domain = domain, message = 'push-pop-level-3'):
                        self.function()
