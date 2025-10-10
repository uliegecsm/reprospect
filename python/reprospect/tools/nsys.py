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
import rich.console
import rich.tree
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
        if nvtx_capture is not None:
            match nvtx_capture:
                case '*':
                    pass
                case _:
                    opts += [
                        '--capture-range=nvtx',
                        f'--capture-range-end={capture_range_end}',
                        f'--nvtx-capture={nvtx_capture}',
                    ]
            opts += ['--trace=nvtx,cuda']
        else:
            opts += ['--trace=cuda']

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
        correlation_src : str = 'correlationId',
        correlation_dst : str = 'correlationId',
    ) -> pandas.Series:
        """
        Select a row from `src`, and return the row from `dst` that matches by correlation ID.
        """
        if isinstance(src, pandas.Series) and selector is None:
            return cls.single_row(data = dst[dst[correlation_dst] == src[correlation_src]])
        else:
            return cls.single_row(data = dst[dst[correlation_dst] == src[selector(src)].squeeze()[correlation_src]])

    class NvtxEvents:
        @typeguard.typechecked
        def __init__(self, events : pandas.DataFrame) -> None:
            self.events = events

        @typeguard.typechecked
        def get(self, accessors : typing.Iterable[str]) -> pandas.DataFrame:
            """
            Find all nested `NVTX` events matching `accessors`.
            """
            if not accessors:
                return self.events

            # Find events matching the first accessor.
            previous = self.events[self.events['text'] == accessors[0]].index

            # For each subsequent accessor, find matching children.
            for accessor in accessors[1:]:
                current = set()
                for idx in previous:
                    for child_idx in self.events.loc[idx, 'children']:
                        if self.events.loc[child_idx, 'text'] == accessor:
                            current.add(child_idx)
                previous = current

            return self.events.iloc[sorted(previous)]

        @typeguard.typechecked
        def to_tree(self) -> rich.tree.Tree:
            """
            Convert to a :py:class:`rich.tree.Tree`.
            """
            @typeguard.typechecked
            def add_branch(*, tree : rich.tree.Tree, nodes : pandas.DataFrame) -> None:
                for _, node in nodes.iterrows():
                    branch = tree.add(f'{node['text']} ({node['eventTypeName']})')
                    if node['children'].any():
                        add_branch(tree = branch, nodes = self.events.loc[node['children']])

            tree = rich.tree.Tree('NVTX events')
            add_branch(tree = tree, nodes = self.events[self.events['level'] == 0])

            return tree

        def __str__(self) -> str:
            """
            Rich representation with :py:meth:`to_tree`.
            """
            with rich.console.Console() as console, console.capture() as capture:
                console.print(self.to_tree(), no_wrap = True)
            return capture.get()

    @functools.cached_property
    @typeguard.typechecked
    def nvtx_events(self) -> 'Report.NvtxEvents':
        """
        Get all `NVTX` events from the `NVTX_EVENTS` table.

        Add a `children` column that contains for each event a list of child indices,
        preserving the hierarchy of the nested `NVTX` ranges.

        Add a `level` column, starting from 0 for the root events.

        .. note::

            Nesting is determined based on `start` and `end` time points.

        .. note::

            Events recorded with registered strings will have there `text` field set to `NULL`,
            and its `textId` field set.

            We correlate the `textId` with the `StringIds` if need be.
        """
        # Get NVTX_EVENTS columns schema metadata, and remove the 'text'.
        # It is the only way to query all columns, while avoiding that the 'COALESCE'
        # duplicates the 'text' row (which would happen if selecting 'NVTX_EVENTS.*').
        columns_but_text = ", ".join(map(
            lambda x: f'NVTX_EVENTS.{x}',
            filter(
                lambda x: x != 'text',
                pandas.read_sql_query("PRAGMA table_info(NVTX_EVENTS);", self.conn)['name']
        )))

        query = f"""
SELECT {columns_but_text},
       COALESCE(NVTX_EVENTS.text, StringIds.value) AS text,
       ENUM_NSYS_EVENT_TYPE.name AS eventTypeName
FROM NVTX_EVENTS
LEFT JOIN ENUM_NSYS_EVENT_TYPE
     ON NVTX_EVENTS.eventType = ENUM_NSYS_EVENT_TYPE.id
LEFT JOIN StringIds
     ON NVTX_EVENTS.textId = StringIds.id
ORDER BY NVTX_EVENTS.start ASC, NVTX_EVENTS.end DESC
"""
        events = pandas.read_sql_query(query, self.conn)

        # Add a 'level' column.
        events['level'] = -1

        # We’ll build parent-child relationships using a stack.
        stack = []
        child_map = {i: [] for i in events.index}

        for idx, event in events.iterrows():
            # Pop any finished parents.
            while stack and not (event['start'] >= events.loc[stack[-1], 'start']
                                and event['end'] <= events.loc[stack[-1], 'end']):
                stack.pop()

            if stack:
                # Current event is a child of the top of the stack.
                parent_idx = stack[-1]
                child_map[parent_idx].append(idx)
                events.loc[idx, 'level'] = events.loc[parent_idx, 'level'] + 1
            else:
                events.loc[idx, 'level'] = 0

            stack.append(idx)

        events['children'] = [pandas.Series(children, dtype = int) for children in child_map.values()]

        return Report.NvtxEvents(events = events)

    @typeguard.typechecked
    def get_events(self, table : str, accessors : typing.Iterable[str]) -> pandas.DataFrame:
        """
        Query all rows in `table` that happen between the `start`/`end` time points
        of the nested `NVTX` range matching `accessors`.

        .. note::

            This replaces `nsys stats` whose `--filter-nvtx` is not powerful enough, as of `Cuda` 13.0.0.
        """
        logging.info(f'Retrieving events in {table} happening within the nested NVTX range {accessors}.')

        filtered = self.nvtx_events.get(accessors = accessors)

        if len(filtered) != 1:
            raise RuntimeError('For now, only one NVTX event is supported.')

        filtered = filtered.squeeze()

        logging.info(f'Events will be filtered in the time frame {filtered['start']} -> {filtered['end']}.')

        query = f"""
SELECT
    {table}.*,
    StringIds.value AS name
FROM {table}
LEFT JOIN StringIds
    ON {table}.nameId = StringIds.id
WHERE {table}.start >= {filtered['start']} AND {table}.end <= {filtered['end']}
ORDER BY {table}.start ASC
        """
        return pandas.read_sql_query(query, self.conn)

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
            hasher.update(shlex.join(map(str, command.args)).encode())

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
