import abc
import typing

import pandas
import rich.console
import rich.table
import rich.tree


def to_string(
    ro: rich.table.Table | rich.tree.Tree,
    *,
    width: int = 200,
    no_wrap: bool = True,
    **kwargs,
) -> str:
    """
    Use :py:class:`rich.console.Console` in capture mode to render a :py:mod:`rich`
    object to a string.
    """
    with rich.console.Console(width=width, **kwargs) as console, console.capture() as capture:
        console.print(ro, no_wrap=no_wrap)
    return capture.get()

def ds_to_table(ds: pandas.Series) -> rich.table.Table:
    """
    Convert a :py:class:`pandas.Series` to a :py:class:`rich.table.Table`.
    """
    rt = rich.table.Table()
    for k in ds.index:
        rt.add_column(str(k))
    rt.add_row(*ds.astype(str))
    return rt

def df_to_table(
    df: pandas.DataFrame,
    *,
    rich_table: rich.table.Table | None = None,
    show_index: bool = False,
) -> rich.table.Table:
    """
    Convert a :py:class:`pandas.DataFrame` to a :py:class:`rich.table.Table`.

    .. note:

        This function is similar to an equivalent function from the `rich-tools` package.

        However, this function allows the indices to be the indices of the :py:class:`pandas.DataFrame`
        rather than an enumeration of the rows.

        There is also an issue with the `rich-tools` function:

        * https://github.com/avi-perl/rich_tools/issues/10
    """
    if rich_table is None:
        rich_table = rich.table.Table()

    if show_index:
        index_name = df.index.name or ''
        rich_table.add_column(str(index_name))

    for column in df.columns:
        rich_table.add_column(str(column))

    for index, value_list in df.iterrows():
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table

def rows_to_table(rows: typing.Iterable[tuple[typing.Any, ...]], *, columns: tuple[str, ...]) -> rich.table.Table:
    """
    Build a :py:class:`rich.table.Table` from `rows`, one table row per tuple.

    Values are converted with :py:class:`str`.
    """
    table = rich.table.Table(*columns)
    for row in rows:
        table.add_row(*map(str, row))
    return table

class TableMixin(metaclass=abc.ABCMeta):
    """
    Define :py:meth:`__str__` based on the :py:class:`rich.table.Table` representation from :py:meth:`to_table`.
    """
    @abc.abstractmethod
    def to_table(self) -> rich.table.Table:
        """
        Convert to a :py:class:`rich.table.Table`.
        """

    @typing.final
    def __str__(self) -> str:
        """
        Use :py:class:`rich.console.Console` in capture mode.
        """
        return to_string(self.to_table())

class TreeMixin(metaclass=abc.ABCMeta):
    """
    Define :py:meth:`__str__` based on the :py:class:`rich.tree.Tree` representation from :py:meth:`to_tree`.
    """
    @abc.abstractmethod
    def to_tree(self) -> rich.tree.Tree:
        """
        Convert to a :py:class:`rich.tree.Tree`.
        """

    @typing.final
    def __str__(self) -> str:
        """
        Use :py:class:`rich.console.Console` in capture mode.
        """
        return to_string(self.to_tree())
