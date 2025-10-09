import copy
import dataclasses
import functools
import json
import logging
import os
import pathlib
import shlex
import shutil
import sqlite3
import subprocess
import typing

import blake3
import pandas
import typeguard

from reprospect.tools import cacher
from reprospect.utils import ldd

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

    @dataclasses.dataclass(frozen = True)
    class Command:
        """
        `nsys` command.
        """
        opts: list[str]
        output: pathlib.Path
        executable: pathlib.Path
        args: list[str]

        @functools.cached_property
        @typeguard.typechecked
        def to_list(self) -> list[str | pathlib.Path]:
            """
            Build the full `nsys` profile command.
            """
            cmd = ['nsys', 'profile']

            cmd += self.opts

            cmd += [
                '--force-overwrite=true',
                f'--output={self.output}',
            ]

            cmd += [self.executable]

            if self.args:
                cmd += self.args

            return cmd

    @typeguard.typechecked
    def get_command(self, *,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_capture : typing.Optional[str] = None,
        capture_range_end : str = 'stop',
        args : typing.Optional[list[str | pathlib.Path]] = None,
    ) -> 'Session.Command':
        """
        Create a :py:class:`Session.Command`.
        """
        opts = [] if opts is None else copy.deepcopy(opts)

        # We want to start data collection when the first NVTX range is met.
        # This reduces the amount of data collected (and makes things faster).
        opts += [
            '--capture-range=nvtx',
            f'--capture-range-end={capture_range_end}',
            f'--nvtx-capture={nvtx_capture}',
            '--trace=nvtx,cuda',
        ] if nvtx_capture else ['--trace=cuda']

        # Disable collecting CPU samples.
        opts += [
            '--sample=none',
            '--backtrace=none',
            '--cpuctxsw=none',
        ]

        return Session.Command(
            opts = opts,
            output = self.output_file_nsys_rep,
            executable = executable,
            args = args,
        )

    @typeguard.typechecked
    def run(
        self,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_capture : typing.Optional[str] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
        cwd : typing.Optional[pathlib.Path] = None,
        env : typing.Optional[typing.MutableMapping] = None,
    ) -> 'Session.Command':
        """
        Run `cmd` with `nsys`.
        """
        command = self.get_command(
            opts = opts,
            nvtx_capture = nvtx_capture,
            executable = executable,
            args = args,
        )

        # For '--capture-range=nvtx' to accept our custom strings, we need to allow unregistered
        # strings to be considered.
        # See https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-interactive-cli-command-sequences.
        if nvtx_capture:
            if env is None:
                env = os.environ.copy()
            env['NSYS_NVTX_PROFILER_REGISTER_ONLY'] = '0'

        logging.info(f"Launching 'nsys' with {command.to_list}.")
        self.output_file_nsys_rep.unlink(missing_ok = True)
        subprocess.check_call(command.to_list, cwd = cwd, env = env)

        return command

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

    @typeguard.typechecked
    def __exit__(self, *args, **kwargs) -> None:
        logging.info(f'Closing connection to {self.db}.')
        self.conn.close()

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
    @staticmethod
    def single_row(*, data : pandas.DataFrame) -> pandas.Series:
        """
        Check that `data` has one row, and squeeze it.
        """
        if len(data) != 1:
             raise RuntimeError(data)
        return data.squeeze()

    @dataclasses.dataclass(frozen = True)
    class PatternSelector:
        """
        A :py:class:`pandas.DataFrame` selector that returns which rows match a regex pattern
        in a specific column.
        """
        pattern : str
        column : str = 'Name'

        @typeguard.typechecked
        def __call__(self, table : pandas.DataFrame) -> pandas.Series:
            return table[self.column].astype(str).str.contains(self.pattern, regex = True)

    @typeguard.typechecked
    @classmethod
    def get_correlated_row(cls, *,
        src : pandas.DataFrame | pandas.Series,
        dst : pandas.DataFrame,
        selector : typing.Optional[typing.Callable[[pandas.DataFrame], pandas.Series]] = None,
    ) -> pandas.Series:
        """
        Select a row from `src`, and return the row from `dst` that matches by correlation ID.
        """
        if isinstance(src, pandas.Series) and selector is None:
            return cls.single_row(data = dst[dst['correlationId'] == src['CorrID']])
        else:
            return cls.single_row(data = dst[dst['correlationId'] == src[selector(src)].squeeze()['CorrID']])

@typeguard.typechecked
def strip_cuda_api_suffix(call : str) -> str:
    """
    Strip suffix like `_v10000` or `_ptsz` from a `Cuda` API `call`.
    """
    return call.split('_')[0]

class Cacher(cacher.Cacher):
    """
    Cacher tailored to `nsys` results.

    `nsys` require quite some time to acquire results.

    On a cache hit, the cacher will serve:
        - `.nsys-rep` file
        - `.sqlite` file

    On a cache miss, `nsys` is launched and the cache entry populated accordingly.

    .. note::

        It is assumed that hashing is faster than running `nsys` itself.

    .. warning::

        The cache should not be shared between machines, since there may be differences between machines
        that influence the results but are not included in the hashing.
    """
    TABLE : str = 'nsys'

    @typeguard.typechecked
    def __init__(self, session : Session, directory : typing.Optional[str | pathlib.Path] = None):
        super().__init__(directory = directory if directory is not None else pathlib.Path(os.environ['HOME']) / '.nsys-cache')
        self.session = session

    @typing.override
    @typeguard.typechecked
    def hash(self, *,
        executable : pathlib.Path,
        opts : typing.Optional[list[str]] = None,
        nvtx_capture : typing.Optional[str] = None,
        args : typing.Optional[list[str | pathlib.Path]] = None,
        env : typing.Optional[typing.MutableMapping] = None,
        **kwargs
    ) -> blake3.blake3:
        """
        Hash based on:
            * `nsys` version
            * `nsys` options (but not the output files)
            * executable content
            * executable arguments
            * linked libraries
            * environment
        """
        hasher = blake3.blake3()

        hasher.update(subprocess.check_output(['nsys', '--version']))

        command = self.session.get_command(
            opts = opts,
            nvtx_capture = nvtx_capture,
            executable = executable,
            args = args,
        )

        if command.opts:
            hasher.update(shlex.join(command.opts).encode())

        hasher.update_mmap(command.executable)

        if command.args:
            hasher.update(shlex.join(command.args).encode())

        for lib in sorted(ldd.get_shared_dependencies(file = command.executable)):
            hasher.update_mmap(lib)

        if env:
            hasher.update(json.dumps(env).encode())

        return hasher

    @typeguard.typechecked
    def populate(self, directory : pathlib.Path, **kwargs) -> Session.Command:
        """
        When there is a cache miss, call :py:meth:`reprospect.tools.nsys.Session.run`.
        Fill the `directory` with the artifacts.
        """
        command = self.session.run(**kwargs)

        shutil.copy(dst = directory, src = command.output)

        return command

    @typeguard.typechecked
    def run(self, **kwargs) -> cacher.Cacher.Entry:
        """
        On a cache hit, copy files from the cache entry.
        """
        entry = self.get(**kwargs)

        if entry.cached:
            shutil.copytree(entry.directory, self.session.output_dir, dirs_exist_ok = True)

        return entry

    @typeguard.typechecked
    def export_to_sqlite(
        self,
        entry : cacher.Cacher.Entry,
        **kwargs,
    ) -> None:
        """
        Export report to `.sqlite`.
        """
        cached = entry.directory / self.session.output_file_sqlite.name

        if cached.is_file():
            logging.info(f'Serving {self.session.output_file_sqlite} from the cache entry {entry}.')
            shutil.copyfile(src = cached, dst = self.session.output_file_sqlite)
        else:
            logging.info(f'Populating the cache entry {entry} with {self.session.output_file_sqlite} from the cache entry {entry}.')
            self.session.export_to_sqlite(**kwargs)
            shutil.copyfile(src = self.session.output_file_sqlite, dst = cached)
