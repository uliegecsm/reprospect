import functools
import logging
import os
import pathlib
import sqlite3
import subprocess
import typing

import pandas
import typeguard

class Session:
    """
    Helper for interacting with an `nsys` session.
    """
    @typeguard.typechecked
    def __init__(self, output_dir : pathlib.Path, output_file_prefix : str) -> None:
        self.output_dir = output_dir
        self.output_file_prefix = output_file_prefix

        self.output_file_nsys_rep = (self.output_dir / self.output_file_prefix).with_suffix('.nsys-rep')
        self.output_file_sqlite   = (self.output_dir / self.output_file_prefix).with_suffix('.sqlite')

    @typeguard.typechecked
    def run(
        self,
        cmd : typing.List[str | pathlib.Path],
        opts : typing.Optional[typing.List[str]] = None,
        nvtx_capture : typing.Optional[str] = None,
        cwd : typing.Optional[pathlib.Path] = None,
        env : typing.Optional[typing.MutableMapping] = None,
    ) -> None:
        """
        Run `cmd` with `nsys`.
        """
        if opts is None: opts = []

        # We want to start data collection when the first NVTX range is met.
        # This reduces the amount of data collected (and makes things faster).
        opts += [
            '--capture-range=nvtx',
            '--capture-range-end=none',
            f'--nvtx-capture={nvtx_capture}',
            '--trace=nvtx,cuda',
        ] if nvtx_capture else ['--trace=cuda']

        run_cmd = [
            'nsys', 'profile',
            # Disable collecting CPU samples.
            '--sample=none',
            '--backtrace=none',
            '--cpuctxsw=none',
            # Output.
            '--force-overwrite=true',
            f'--output={self.output_file_nsys_rep}',
        ] + opts + cmd

        # For '--capture-range=nvtx' to accept our custom strings, we need to allow unregistered
        # strings to be considered.
        # See https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-interactive-cli-command-sequences.
        if nvtx_capture:
            env['NSYS_NVTX_PROFILER_REGISTER_ONLY'] = '0'

        logging.info(f"Launching 'nsys' with {run_cmd}.")
        self.output_file_nsys_rep.unlink(missing_ok = True)
        subprocess.check_call(run_cmd, cwd = cwd, env = env)

    @typeguard.typechecked
    def export_to_sqlite(
        self,
        cwd : pathlib.Path = pathlib.Path.cwd(),
    ) -> None:
        """
        Export report to `.sqlite`.
        """
        cmd = [
            'nsys', 'stats',
            '--force-overwrite=true',
            '--force-export=true',
            f'--sqlite={self.output_file_sqlite}',
            self.output_file_nsys_rep,
        ]

        logging.info(f"Exporting to 'sqlite' with {cmd}.")
        self.output_file_sqlite.unlink(missing_ok = True)
        subprocess.check_call(cmd, cwd = cwd)

    @typeguard.typechecked
    def extract_statistical_report(
        self,
        report : str = 'cuda_api_sum',
        filter_nvtx : typing.Optional[str] = None,
        cwd : pathlib.Path = pathlib.Path.cwd(),
    ) -> pandas.DataFrame:
        """
        Extract the `Cuda` `API` call stats, filtering the database with `filter_nvtx`.
        """
        cmd = [
            'nsys', 'stats',
            f'--output={self.output_dir / self.output_file_prefix}',
            f'--report={report}',
            '--format=csv',
            '--timeunit=usec',
        ]
        if filter_nvtx:
            cmd += ['--filter-nvtx=' + filter_nvtx]
        cmd += [self.output_file_sqlite]

        # 'nsys stats' will output to a file whose name follows the convention
        #    <basename>_<analysis&args>.<output_format>
        suffix = '_nvtx=' + filter_nvtx.replace('/', '-') if filter_nvtx else ''
        output = self.output_dir / f'{self.output_file_prefix}_{report}{suffix}.csv'

        logging.info(f'Removing {output} and extracting statistical report \'{report}\' from {self.output_file_sqlite} with {cmd}.')
        output.unlink(missing_ok = True)
        subprocess.check_call(cmd, cwd = cwd)

        return pandas.read_csv(output)

class Report:
    """
    Helper for reading the `SQLite` export of a `nsys` report.
    """
    @typeguard.typechecked
    def __init__(self, *, db : pathlib.Path) -> None:
        self.db = db

    @typeguard.typechecked
    def __enter__(self) -> 'Report':
        logging.info(f'Connecting to {self.db}.')
        self.conn = sqlite3.connect(self.db)
        return self

    @functools.cached_property
    @typeguard.typechecked
    def tables(self) -> list[str]:
        """
        Tables in the report.
        """
        logging.info(f'Listing tables in {self.db}.')
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    @typeguard.typechecked
    def table(self, *, name : str) -> pandas.DataFrame:
        """
        Get a table from the report.
        """
        logging.info(f'Retrieving table {name} in {self.db}.')
        return pandas.read_sql_query(f"SELECT * FROM {name};", self.conn)

    @typeguard.typechecked
    def __exit__(self, *args, **kwargs) -> None:
        logging.info(f'Closing connection to {self.db}.')
        self.conn.close()

@typeguard.typechecked
def strip_cuda_api_suffix(call : str) -> str:
    """
    Strip suffix like `_v10000` or `_ptsz` from a `Cuda` API `call`.
    """
    return call.split('_')[0]
