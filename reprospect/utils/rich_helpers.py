import abc
import typing

import pandas
import rich.console
import rich.table
import rich.tree
import rich_tools

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
    with rich.console.Console(width = width, **kwargs) as console, console.capture() as capture:
        console.print(ro, no_wrap = no_wrap)
    return capture.get()

def ds_to_table(ds: pandas.Series) -> rich.table.Table:
    """
    Convert a :py:class:`pandas.Series` to a :py:class:`rich.table.Table`.
    """
    rt = rich.table.Table()
    for k in ds.index:
        rt.add_column(str(k))
    rt.add_row(*(str(v) for v in ds.values))
    return rt

def df_to_table(
    df: pandas.DataFrame,
    *,
    rich_table: rich.table.Table | None = None,
    show_index: bool = False,
    **kwargs,
) -> rich.table.Table:
    """
    Convert a :py:class:`pandas.DataFrame` to a :py:class:`rich.table.Table`.

    .. note:

        This wrapper around the equivalent function from the `rich-tools` package
        can be avoided once an issue with their function is resolved:

        * https://github.com/avi-perl/rich_tools/issues/10
    """
    if rich_table is None:
        rich_table = rich.table.Table()

    return rich_tools.df_to_table(df, rich_table = rich_table, show_index = show_index, **kwargs)

class TableMixin(metaclass = abc.ABCMeta):
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

class TreeMixin(metaclass = abc.ABCMeta):
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
