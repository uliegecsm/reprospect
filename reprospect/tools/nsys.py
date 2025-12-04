import dataclasses
import functools
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import typing

import attrs
import blake3
import pandas
import rich.tree

from reprospect.tools import cacher
from reprospect.utils import ldd
from reprospect.utils import rich_helpers

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

@attrs.define(frozen = True, slots = True, kw_only = True)
class Command: # pylint: disable=too-many-instance-attributes, duplicate-code
    """
    Run a ``nsys`` command line.
    """
    executable: str | pathlib.Path
    """Executable to run."""
    output: pathlib.Path
    """Report file."""
    opts : tuple[str, ...] = ()
    """Options that do not involve paths."""
    nvtx_capture : str | None = None
    """NVTX capture string."""
    capture_range_end : str = 'stop'
    """NVTX capture range end."""
    args: tuple[str | pathlib.Path, ...] | None = None
    """Arguments to pass to the executable."""
    env : typing.Mapping[str, str] | None = None
    """Mapping used to update the environment before running, see :py:meth:`run`."""

    cmd : tuple[str | pathlib.Path, ...] = attrs.field(init = False)

    def __attrs_post_init__(self) -> None:
        """
        Enrich :py:attr:`opts` and build :py:attr:`cmd`.
        """
        if self.output.suffix != '.nsys-rep':
            object.__setattr__(self, 'output', self.output.parent / f'{self.output.name}.nsys-rep')

        # Disable collecting CPU samples.
        opts : tuple[str, ...] = self.opts + (
            '--sample=none',
            '--backtrace=none',
            '--cpuctxsw=none',
        )

        # We want to start data collection when the first NVTX range is met.
        # This reduces the amount of data collected (and makes things faster).
        if self.nvtx_capture is not None:
            match self.nvtx_capture:
                case '*':
                    pass
                case _:
                    opts += (
                        '--capture-range=nvtx',
                        f'--capture-range-end={self.capture_range_end}',
                        f'--nvtx-capture={self.nvtx_capture}',
                    )
            opts += ('--trace=nvtx,cuda',)
        else:
            opts += ('--trace=cuda',)

        object.__setattr__(self, 'opts', opts)

        # Build the final full command.
        object.__setattr__(self, 'cmd', (
            'nsys', 'profile',
            *self.opts,
            '--force-overwrite=true', '-o', self.output,
            self.executable,
            *(self.args or ()),
        ))

    def run(self, *,
        cwd : pathlib.Path | None = None,
        env : typing.MutableMapping[str, str] | None = None,
    ) -> int:
        if (self.nvtx_capture is not None or self.env is not None) and env is None:
            env = os.environ.copy()

        # For '--capture-range=nvtx' to accept our custom strings, we need to allow unregistered
        # strings to be considered.
        # See https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-interactive-cli-command-sequences.
        if self.nvtx_capture:
            assert env is not None
            env['NSYS_NVTX_PROFILER_REGISTER_ONLY'] = '0'

        if self.env:
            assert env is not None
            env.update(self.env)

        self.output.unlink(missing_ok = True)
        return subprocess.check_call(args = self.cmd, env = env, cwd = cwd)

@dataclasses.dataclass(frozen = True, slots = True)
class Session:
    """
    `Nsight Systems` session interface.
    """
    command : Command

    def run(
        self,
        cwd : pathlib.Path | None = None,
        env : typing.MutableMapping[str, str] | None = None,
    ) -> None:
        """
        Run ``nsys`` using :py:attr:`command`.
        """
        logging.info(f"Launching 'nsys' with {self.command.cmd}.")
        self.command.run(cwd = cwd, env = env)

    def export_to_sqlite(
        self,
        cwd : pathlib.Path | None = None,
    ) -> pathlib.Path:
        """
        Export report to ``.sqlite``.
        """
        output_file_sqlite = self.command.output.with_suffix('.sqlite')

        cmd : tuple[str | pathlib.Path, ...] = (
            'nsys', 'export',
            '--type', 'sqlite',
            f'--output={output_file_sqlite}',
            self.command.output,
        )

        logging.info(f"Exporting to 'sqlite' with {cmd}.")
        output_file_sqlite.unlink(missing_ok = True)
        subprocess.check_call(cmd, cwd = cwd)

        return output_file_sqlite

    def extract_statistical_report(
        self,
        report : str = 'cuda_api_sum',
        filter_nvtx : str | None = None,
        cwd : pathlib.Path | None = None,
    ) -> pandas.DataFrame:
        """
        Extract `report`, filtering the database with `filter_nvtx`.
        """
        output_file_sqlite = self.command.output.with_suffix('.sqlite')

        cmd : tuple[str | pathlib.Path, ...] = (
            'nsys', 'stats',
            f'--output={self.command.output.parent / self.command.output.stem}',
            f'--report={report}',
            '--format=csv',
            '--timeunit=usec',
            *(('--filter-nvtx=' + filter_nvtx,) if filter_nvtx else ()),
            output_file_sqlite,
        )

        # 'nsys stats' will output to a file whose name follows the convention
        #    <basename>_<analysis&args>.<output_format>
        suffix = '_nvtx=' + filter_nvtx.replace('/', '-') if filter_nvtx else ''
        output_file_csv = self.command.output.parent / f'{self.command.output.stem}_{report}{suffix}.csv'

        logging.info(f'Extracting statistical report \'{report}\' from {output_file_sqlite} with {cmd}.')
        output_file_csv.unlink(missing_ok = True)
        subprocess.check_call(cmd, cwd = cwd)
        assert output_file_csv.is_file()

        return pandas.read_csv(output_file_csv)

@dataclasses.dataclass(frozen = True, slots = True)
class ReportPatternSelector:
    """
    A :py:class:`pandas.DataFrame` selector that returns which rows match a regex pattern
    in a specific column.
    """
    pattern : str | re.Pattern[str]
    column : str = 'Name'

    def __call__(self, table : pandas.DataFrame) -> pandas.Series:
        return table[self.column].astype(str).str.contains(self.pattern, regex = True)

class ReportNvtxEvents(rich_helpers.TreeMixin):
    def __init__(self, events : pandas.DataFrame) -> None:
        self.events = events

    def get(self, accessors : typing.Sequence[str]) -> pandas.DataFrame:
        """
        Find all nested NVTX events matching `accessors`.
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

    @override
    def to_tree(self) -> rich.tree.Tree:
        def add_branch(*, tree : rich.tree.Tree, nodes : pandas.DataFrame) -> None:
            for _, node in nodes.iterrows():
                branch = tree.add(f"{node['text']} ({node['eventTypeName']})")
                if node['children'].any():
                    add_branch(tree = branch, nodes = self.events.loc[node['children']])

        tree = rich.tree.Tree('NVTX events')
        add_branch(tree = tree, nodes = self.events[self.events['level'] == 0])

        return tree

class Report:
    """
    Helper for reading the `SQLite` export of a ``nsys`` report.
    """
    def __init__(self, *, db : pathlib.Path) -> None:
        self.db = db
        self.conn : sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        logging.info(f'Connecting to {self.db}.')
        self.conn = sqlite3.connect(self.db)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        logging.info(f'Closing connection to {self.db}.')
        if self.conn is not None:
            self.conn.close()

    @functools.cached_property
    def tables(self) -> list[str]:
        """
        Tables in the report.
        """
        logging.info(f'Listing tables in {self.db}.')
        assert self.conn is not None
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    def table(self, *, name : str, **kwargs) -> pandas.DataFrame:
        """
        Get a table from the report.
        """
        logging.info(f'Retrieving table {name} in {self.db}.')
        return pandas.read_sql_query(f"SELECT * FROM {name};", self.conn, **kwargs)

    @staticmethod
    def single_row(*, data : pandas.DataFrame) -> pandas.Series:
        """
        Check that `data` has one row, and squeeze it.
        """
        if len(data) != 1:
            raise RuntimeError(data)
        return data.squeeze()

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
        if isinstance(src, pandas.DataFrame) and selector is not None:
            return cls.single_row(data = dst[dst[correlation_dst] == src[selector(src)].squeeze()[correlation_src]])
        raise RuntimeError()

    @classmethod
    def get_correlated_rows(cls, *,
        src : pandas.DataFrame | pandas.Series,
        dst : pandas.DataFrame,
        selector : typing.Optional[typing.Callable[[pandas.DataFrame], pandas.Series]] = None,
        correlation_src : str = 'correlationId',
        correlation_dst : str = 'correlationId',
    ) -> pandas.DataFrame:
        """
        Similar to :py:meth:`get_correlated_row`, but *may* match more than one row.
        """
        if isinstance(src, pandas.Series) and selector is None:
            return dst[dst[correlation_dst] == src[correlation_src]]
        if isinstance(src, pandas.DataFrame) and selector is not None:
            return dst[dst[correlation_dst] == src[selector(src)].squeeze()[correlation_src]]
        raise RuntimeError()

    @functools.cached_property
    def nvtx_events(self) -> ReportNvtxEvents:
        """
        Get all NVTX events from the `NVTX_EVENTS` table.

        Add a `children` column that contains for each event a list of child indices,
        preserving the hierarchy of the nested NVTX ranges.

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
        stack : list[typing.Hashable] = []
        child_map : dict[typing.Hashable, list[typing.Hashable]] = {i: [] for i in events.index}

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

        return ReportNvtxEvents(events = events)

    def get_events(self, table : str, accessors : typing.Sequence[str], stringids : typing.Optional[str] = 'nameId') -> pandas.DataFrame:
        """
        Query all rows in `table` that happen between the `start`/`end` time points
        of the nested NVTX range matching `accessors`.

        :param stringids: Some tables have a column to be correlated with the `StringIds` table.

        .. note::

            This replaces `nsys stats` whose `--filter-nvtx` is not powerful enough, as of CUDA 13.0.0.
        """
        logging.info(f'Retrieving events in {table} happening within the nested NVTX range {accessors}.')

        filtered = self.nvtx_events.get(accessors = accessors)

        if len(filtered) != 1:
            raise RuntimeError(f'Expecting exactly one NVTX event, got {len(filtered)}.')

        filtered = filtered.squeeze()

        logging.info(f"Events will be filtered in the time frame {filtered['start']} -> {filtered['end']}.")

        select = [f'{table}.*']
        join   = []

        if stringids is not None:
            select.append('StringIds.value AS name')
            join  .append(f'LEFT JOIN StringIds ON {table}.{stringids} = StringIds.id')

        query = f"""
SELECT {', '.join(select)}
FROM {table}
{' '.join(join)}
WHERE {table}.start >= {filtered['start']} AND {table}.end <= {filtered['end']}
ORDER BY {table}.start ASC
"""
        return pandas.read_sql_query(query, self.conn)

def strip_cuda_api_suffix(call : str) -> str:
    """
    Strip suffix like `_v10000` or `_ptsz` from a CUDA API `call`.
    """
    return call.split('_')[0]

class Cacher(cacher.Cacher):
    """
    Cacher tailored to ``nsys`` results.

    ``nsys`` runs require quite some time to acquire results.

    On a cache hit, the cacher will serve:

    - ``.nsys-rep`` file
    - ``.sqlite`` file

    On a cache miss, ``nsys`` is launched and the cache entry populated accordingly.

    .. note::

        It is assumed that hashing is faster than running ``nsys`` itself.

    .. warning::

        The cache should not be shared between machines, since there may be differences between machines
        that influence the results but are not included in the hashing.
    """
    TABLE : typing.ClassVar[str] = 'nsys'

    def __init__(self, *, directory : str | pathlib.Path | None = None):
        super().__init__(directory = directory or (pathlib.Path(os.environ['HOME']) / '.nsys-cache'))

    def hash_impl(self, *, command : Command) -> blake3.blake3:
        """
        Hash based on:

        * ``nsys`` version
        * ``nsys`` options (but not the output files)
        * executable content
        * executable arguments
        * linked libraries
        * environment
        """
        hasher = blake3.blake3() # pylint: disable=not-callable

        hasher.update(subprocess.check_output(('nsys', '--version')))

        if command.opts:
            hasher.update(shlex.join(command.opts).encode())

        hasher.update_mmap(command.executable)

        if command.args:
            hasher.update(shlex.join(map(str, command.args)).encode())

        if command.env:
            hasher.update(json.dumps(command.env).encode())

        for lib in sorted(ldd.get_shared_dependencies(file = command.executable)):
            hasher.update_mmap(lib)

        return hasher

    @override
    def hash(self, **kwargs) -> blake3.blake3:
        return self.hash_impl(command = kwargs['command'])

    @override
    def populate(self, directory : pathlib.Path, **kwargs) -> None:
        """
        When there is a cache miss, call :py:meth:`reprospect.tools.nsys.Session.run`.
        Fill the `directory` with the artifacts.
        """
        command = kwargs.pop('command')

        Session(command = command).run(**kwargs)

        shutil.copy(dst = directory, src = command.output)

    def run(self, command : Command, **kwargs) -> cacher.Cacher.Entry:
        """
        On a cache hit, copy files from the cache entry.
        """
        entry = self.get(command = command, **kwargs)

        if entry.cached:
            shutil.copytree(entry.directory, command.output.parent, dirs_exist_ok = True)

        return entry

    @staticmethod
    def export_to_sqlite(
        command : Command,
        entry : cacher.Cacher.Entry,
        **kwargs,
    ) -> pathlib.Path:
        """
        Export report to ``.sqlite``.
        """
        output_file_sqlite = command.output.with_suffix('.sqlite')

        cached = entry.directory / output_file_sqlite.name

        if cached.is_file():
            logging.info(f'Serving {output_file_sqlite} from the cache entry {entry}.')
            shutil.copyfile(src = cached, dst = output_file_sqlite)
        else:
            logging.info(f'Populating the cache entry {entry} with {output_file_sqlite} from the cache entry {entry}.')
            Session(command = command).export_to_sqlite(**kwargs)
            shutil.copyfile(src = output_file_sqlite, dst = cached)

        return output_file_sqlite
