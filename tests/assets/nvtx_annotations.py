import contextlib
import logging

import nvtx


@contextlib.contextmanager
def push_pop_range(domain: nvtx.Domain, **kwargs):
    domain.push_range(attributes=domain.get_event_attributes(**kwargs))
    try:
        yield
    finally:
        domain.pop_range()

@contextlib.contextmanager
def start_end_range(domain: nvtx.Domain, **kwargs):
    range_id = domain.start_range(attributes=domain.get_event_attributes(**kwargs))
    try:
        yield
    finally:
        domain.end_range(range_id)

class TestNVTX:
    """
    Use the `Python` interface of NVTX to create intricate situations.

    References:

    * https://nvidia.github.io/NVTX/python/
    """
    @classmethod
    def function(cls) -> None:
        logging.info(cls)

    def test_sanity(self):
        assert nvtx.enabled()

    def test_context_managed_annotation(self, request) -> None:
        """
        Context-managed annotation.
        """
        with nvtx.annotate(message=request.node.name, color='green'):
            self.function()

    def test_push_pop_range_in_domain(self, request) -> None:
        """
        Push/pop range in a domain.
        """
        with push_pop_range(domain=nvtx.get_domain(name=request.node.name), message='osef'):
            self.function()

    def test_start_end_range_in_domain(self, request) -> None:
        """
        Start/end range in a domain.
        """
        with start_end_range(domain=nvtx.get_domain(name=request.node.name), message='osef'):
            self.function()

    def test_registered_string(self, request) -> None:
        """
        Registered string.
        """
        domain = nvtx.get_domain(name=request.node.name)

        reg = domain.get_registered_string(string='my very long string that is better made a registered string')

        assert reg is not None

    @classmethod
    def intricate(cls, domain) -> None:
        """
        Build a situation with many intricate ranges.
        """
        with start_end_range(domain=domain, message='start-end-level-0'): # noqa: SIM117
            with push_pop_range(domain=domain, message='push-pop-level-1'):
                with push_pop_range(domain=domain, message='push-pop-level-2'):
                    for _ in range(3):
                        with push_pop_range(domain=domain, message='push-pop-level-3'):
                            cls.function()

    def test_intricate(self, request) -> None:
        """
        Build a situation with many intricate ranges.
        """
        self.intricate(domain=nvtx.get_domain(name=request.node.name))

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    TestNVTX.intricate(domain=nvtx.get_domain('intricate'))
