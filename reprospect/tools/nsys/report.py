import dataclasses
import functools
import logging
import pathlib
import re
import sqlite3
import sys
import typing

import pandas
import rich.tree

from reprospect.utils import rich_helpers

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

@dataclasses.dataclass(frozen=True, slots=True)
class ReportPatternSelector:
    """
    A :py:class:`pandas.DataFrame` selector that returns which rows match a regex pattern
    in a specific column.
    """
    pattern: str | re.Pattern[str]
    column: str = 'Name'

    def __call__(self, table: pandas.DataFrame) -> pandas.Series:
        return table[self.column].astype(str).str.contains(self.pattern, regex=True)

class ReportNvtxEvents(rich_helpers.TreeMixin):
    def __init__(self, events: pandas.DataFrame) -> None:
        self.events = events

    def get(self, accessors: typing.Sequence[str]) -> pandas.DataFrame:
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
        def add_branch(*, tree: rich.tree.Tree, nodes: pandas.DataFrame) -> None:
            for _, node in nodes.iterrows():
                branch = tree.add(f"{node['text']} ({node['eventTypeName']})")
                if node['children'].any():
                    add_branch(tree=branch, nodes=self.events.loc[node['children']])

        tree = rich.tree.Tree('NVTX events')
        add_branch(tree=tree, nodes=self.events[self.events['level'] == 0])

        return tree

class Report:
    """
    Helper for reading the `SQLite` export of a ``nsys`` report.
    """
    def __init__(self, *, db: pathlib.Path) -> None:
        self.db = db
        self.conn: sqlite3.Connection | None = None

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

    def table(self, *, name: str, **kwargs) -> pandas.DataFrame:
        """
        Get a table from the report.
        """
        logging.info(f'Retrieving table {name} in {self.db}.')
        return pandas.read_sql_query(f"SELECT * FROM {name};", self.conn, **kwargs)

    @staticmethod
    def single_row(*, data: pandas.DataFrame) -> pandas.Series:
        """
        Check that `data` has one row, and squeeze it.
        """
        if len(data) != 1:
            raise RuntimeError(data)
        return data.squeeze()

    @classmethod
    def get_correlated_row(cls, *,
        src: pandas.DataFrame | pandas.Series,
        dst: pandas.DataFrame,
        selector: typing.Callable[[pandas.DataFrame], pandas.Series] | None = None,
        correlation_src: str = 'correlationId',
        correlation_dst: str = 'correlationId',
    ) -> pandas.Series:
        """
        Select a row from `src`, and return the row from `dst` that matches by correlation ID.
        """
        if isinstance(src, pandas.Series) and selector is None:
            return cls.single_row(data=dst[dst[correlation_dst] == src[correlation_src]])
        if isinstance(src, pandas.DataFrame) and selector is not None:
            return cls.single_row(data=dst[dst[correlation_dst] == src[selector(src)].squeeze()[correlation_src]])
        raise RuntimeError

    @classmethod
    def get_correlated_rows(cls, *,
        src: pandas.DataFrame | pandas.Series,
        dst: pandas.DataFrame,
        selector: typing.Callable[[pandas.DataFrame], pandas.Series] | None = None,
        correlation_src: str = 'correlationId',
        correlation_dst: str = 'correlationId',
    ) -> pandas.DataFrame:
        """
        Similar to :py:meth:`get_correlated_row`, but *may* match more than one row.
        """
        if isinstance(src, pandas.Series) and selector is None:
            return dst[dst[correlation_dst] == src[correlation_src]]
        if isinstance(src, pandas.DataFrame) and selector is not None:
            return dst[dst[correlation_dst] == src[selector(src)].squeeze()[correlation_src]]
        raise RuntimeError

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
        columns_but_text = ', '.join(
            f'NVTX_EVENTS.{x}'
            for x in pandas.read_sql_query("PRAGMA table_info(NVTX_EVENTS);", self.conn)['name']
            if x != 'text'
        )

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

        # We'll build parent-child relationships using a stack.
        stack: list[typing.Hashable] = []
        child_map: dict[typing.Hashable, list[typing.Hashable]] = {i: [] for i in events.index}

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

        events['children'] = [pandas.Series(children, dtype=int) for children in child_map.values()]

        return ReportNvtxEvents(events=events)

    def get_events(self, table: str, accessors: typing.Sequence[str], stringids: str | None = 'nameId') -> pandas.DataFrame:
        """
        Query all rows in `table` that happen between the `start`/`end` time points
        of the nested NVTX range matching `accessors`.

        :param stringids: Some tables have a column to be correlated with the `StringIds` table.

        .. note::

            This replaces `nsys stats` whose `--filter-nvtx` is not powerful enough, as of CUDA 13.0.0.
        """
        logging.info(f'Retrieving events in {table} happening within the nested NVTX range {accessors}.')

        filtered = self.nvtx_events.get(accessors=accessors)

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

def strip_cuda_api_suffix(call: str) -> str:
    """
    Strip suffix like `_v10000` or `_ptsz` from a CUDA API `call`.
    """
    return call.split('_', maxsplit=1)[0]
