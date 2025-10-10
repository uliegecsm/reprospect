import io
import logging
import pprint
import subprocess
import typing

import pandas
import rich.table
import typeguard

class Manager:
    """
    Manage clock rates of `NVIDIA` GPUs.

    .. note::

        It requires `nvidia-smi` and appropriate permissions.

    References:
        * https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate
    """
    @typeguard.typechecked
    def __init__(self, devices : typing.Optional[list[int]] = None) -> None:
        """
        :param devices: Device IDs that will be managed.
        """
        gpus = self.retrieve_table_from(cmd = ['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv'])
        logging.info(f'Detected GPUs:\n{pprint.pformat(gpus)}')

        self.devices = gpus if devices is None else gpus[gpus['index'].astype(int).isin(devices)]

    @staticmethod
    @typeguard.typechecked
    def retrieve_table_from(cmd : list[str]) -> pandas.DataFrame:
        """
        Retrieve a table from `nvidia-smi`.
        """
        return pandas.read_csv(
            io.StringIO(subprocess.check_output(cmd).decode()),
            sep = ',',
            skipinitialspace = True,
        )

    @typeguard.typechecked
    def get_current_clocks(self) -> pandas.DataFrame:
        """
        Query current GPU clock information.
        """
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,clocks.current.sm,clocks.current.memory,pstate,power.draw',
            '--format=csv'
        ]

        for device in self.devices['index']:
            cmd.append(f'--id={device}')

        clocks = self.retrieve_table_from(cmd = cmd)

        for col in ['clocks.current.sm [MHz]', 'clocks.current.memory [MHz]']:
            clocks[col] = clocks[col].str.extract(r'(\d+)').astype(int)

        return clocks

    @typeguard.typechecked
    def get_supported_clocks(self) -> dict[int, pandas.DataFrame]:
        """
        Query all supported clock combinations.
        """
        cmd = [
            'nvidia-smi',
            '--query-supported-clocks=gpu_uuid,memory,graphics',
            '--format=csv'
        ]

        for device in self.devices['index']:
            cmd.append(f'--id={device}')

        supported = self.retrieve_table_from(cmd = cmd)

        clocks = {}

        for _, device in self.devices.iterrows():
            clocks[device['index']] = supported[supported['gpu_uuid'] == device['uuid']]
            for col in ['memory [MHz]', 'graphics [MHz]']:
                clocks[device['index']][col] = clocks[device['index']][col].str.extract(r'(\d+)').astype(int)

        return clocks

    @staticmethod
    @typeguard.typechecked
    def to_table(data : pandas.DataFrame, **kwargs) -> rich.table.Table:
        """
        Convert a :py:class:`pandas.DataFrame` to a :py:class:`rich.table.Table`.
        """
        table = rich.table.Table(**kwargs)

        for column in data.columns:
            table.add_column(str(column))

        for _, row in data.iterrows():
            table.add_row(*map(str, row.tolist()))

        return table

    @typeguard.typechecked
    def get_max_clocks(self) -> pandas.DataFrame:
        """
        For each device, find the supported clock rates with highest values.
        It is supposed to be the "fastest" operating state.

        .. note::

            It assumes that there is a dominant point (*i.e.* it is not a pareto front).
        """
        supported_clocks = self.get_supported_clocks()

        return pandas.concat({
            idx: clocks[
                (clocks["memory [MHz]"] == clocks["memory [MHz]"].max()) &
                (clocks["graphics [MHz]"] == clocks["graphics [MHz]"].max())
            ]
            for idx, clocks in enumerate(supported_clocks.values())
        })

    @typeguard.typechecked
    def lock(self, clocks : pandas.DataFrame) -> None:
        """
        Lock GPU clocks to specified rates.
        """
        for _, device in self.devices.iterrows():
            clock = clocks[clocks['gpu_uuid'] == device['uuid']].squeeze()
            logging.info(f'Locking clocks of device {device['index']} to {clock}.')
            gpu = clock['graphics [MHz]']
            mem = clock['memory [MHz]']
            subprocess.check_call(['nvidia-smi', f'--id={device['index']}', f'--lock-gpu-clocks={gpu},{gpu}'])
            subprocess.check_call(['nvidia-smi', f'--id={device['index']}', f'--lock-memory-clocks={mem},{mem}'])

    @typeguard.typechecked
    def reset(self):
        """
        Reset GPU clocks to default (auto) mode.
        """
        for device in self.devices['index']:
            logging.info(f'Reset clocks of device {device}.')
            subprocess.check_call(['nvidia-smi', f'--id={device}', '--reset-gpu-clocks'])
            subprocess.check_call(['nvidia-smi', f'--id={device}', '--reset-memory-clocks'])

class Locker:
    """
    Context-managed locking.
    """
    @typeguard.typechecked
    def __init__(self, manager : Manager, clocks : pandas.DataFrame) -> None:
        self.manager = manager
        self.clocks  = clocks

    @typeguard.typechecked
    def __enter__(self) -> 'Locker':
        """
        Lock to :py:attr:`clocks`.
        """
        self.manager.lock(clocks = self.clocks)
        return self

    @typeguard.typechecked
    def __exit__(self, *args, **kwargs) -> None:
        """
        Reset.
        """
        self.manager.reset()
