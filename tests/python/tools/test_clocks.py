import logging
import pathlib

import matplotlib.pyplot
import pytest
import rich.console
import typeguard

from reprospect.tools import clocks

@pytest.fixture(scope = 'class')
@typeguard.typechecked
def console() -> rich.console.Console:
    return rich.console.Console()

@pytest.fixture(scope = 'function')
@typeguard.typechecked
def manager() -> clocks.Manager:
    manager = clocks.Manager()
    assert len(manager.devices) >= 1
    return manager

class TestManager:
    """
    Test :py:class:`reprospect.tools.clocks.Manager`.
    """
    def test_init(self) -> None:
        """
        Check initialization of the list of devices.
        """
        manager = clocks.Manager(devices = [0])

        assert len(manager.devices) == 1

    def test_get_current_clocks(self, manager, console) -> None:
        """
        Test :py:meth:`reprospect.tools.clocks.Manager.get_current_clocks`.
        """
        state = manager.get_current_clocks()
        assert len(state) == len(manager.devices)

        assert state['clocks.current.sm [MHz]'    ].dtype == int
        assert state['clocks.current.memory [MHz]'].dtype == int

        console.log(manager.to_table(data = state))

    def test_get_supported_clocks(self, manager, console) -> None:
        """
        Test :py:meth:`reprospect.tools.clocks.Manager.get_supported_clocks`.
        """
        supported = manager.get_supported_clocks()
        assert len(supported) == len(manager.devices)

        for table in supported.values():
            assert table['memory [MHz]'  ].dtype == int
            assert table['graphics [MHz]'].dtype == int

        for table in supported.values():
            console.log(manager.to_table(data = table))

        fig, axes = matplotlib.pyplot.subplots(nrows = 1, ncols = len(manager.devices), figsize = (15, 10), squeeze = False)

        ON_X = 'graphics [MHz]'
        ON_Y = 'memory [MHz]'

        for device, clocks in supported.items():
            axes[0, device].plot(clocks[ON_X], clocks[ON_Y], 'o')
            axes[0, device].set_xlabel(ON_X)
            axes[0, device].set_ylabel(ON_Y)
            axes[0, device].tick_params(axis = 'x', labelrotation = 45)
            axes[0, device].xaxis.set_major_locator(matplotlib.pyplot.MaxNLocator(15))

        fname = pathlib.Path.cwd() / 'supported_clocks.svg'
        logging.info(f'Exporting supported clocks to {fname}.')
        fig.savefig(fname = fname, bbox_inches = 'tight', transparent = False)

    def test_get_max_clocks(self, manager, console) -> None:
        """
        Test :py:meth:`reprospect.tools.clocks.Manager.get_max_clocks`.
        """
        clocks = manager.get_max_clocks()
        assert len(clocks) == len(manager.devices)

        assert clocks['memory [MHz]'  ].dtype == int
        assert clocks['graphics [MHz]'].dtype == int

        console.log(manager.to_table(data = clocks))

    def test_lock(self, manager, console) -> None:
        """
        Test :py:meth:`reprospect.tools.clocks.Manager.lock`.
        """
        clocks = manager.get_max_clocks()

        try:
            console.log(manager.to_table(data = clocks, title = 'Detected max clocks'))

            console.log(manager.to_table(data = manager.get_current_clocks(), title = 'Current state before locking the clocks'))

            manager.lock(clocks = clocks)

            console.log(manager.to_table(data = manager.get_current_clocks(), title = 'Current state after locking the clocks'))
        finally:
            manager.reset()

class TestLocker:
    """
    Tests for :py:class:`reprospect.tools.clocks.Locker`.
    """
    def test(self, manager, console) -> None:
        """
        Enter the context, and exit.
        """
        with clocks.Locker(manager = manager, clocks = manager.get_max_clocks()) as locker:
            console.log(manager.to_table(data = manager.get_current_clocks(), title = 'Current state within the context'))
        console.log(manager.to_table(data = manager.get_current_clocks(), title = 'Current state outside the context'))
