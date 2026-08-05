import pandas as pd
import pytest

from reprospect.utils import rich_helpers


@pytest.fixture(scope='session')
def df() -> pd.DataFrame:
    return pd.DataFrame(
        data=[(1, 'a'), (2, 'b'), (3, 'c')],
        index=['row1', 'row2', 'row3'],
        columns=['col1', 'col2'],
    )

def test_df_to_table_do_not_show_index(df: pd.DataFrame) -> None:
    """
    Test :py:func:`reprospect.utils.rich_helpers.df_to_table` with `show_index=False`.
    """
    assert rich_helpers.to_string(rich_helpers.df_to_table(df, show_index=False)) == """\
┏━━━━━━┳━━━━━━┓
┃ col1 ┃ col2 ┃
┡━━━━━━╇━━━━━━┩
│ 1    │ a    │
│ 2    │ b    │
│ 3    │ c    │
└──────┴──────┘
"""

def test_df_to_table_show_index(df: pd.DataFrame) -> None:
    """
    Test :py:func:`reprospect.utils.rich_helpers.df_to_table` with `show_index=True`.
    """
    assert rich_helpers.to_string(rich_helpers.df_to_table(df, show_index=True)) == """\
┏━━━━━━┳━━━━━━┳━━━━━━┓
┃      ┃ col1 ┃ col2 ┃
┡━━━━━━╇━━━━━━╇━━━━━━┩
│ row1 │ 1    │ a    │
│ row2 │ 2    │ b    │
│ row3 │ 3    │ c    │
└──────┴──────┴──────┘
"""

def test_rows_to_table() -> None:
    """
    Test :py:func:`reprospect.utils.rich_helpers.rows_to_table`.
    """
    rows = [
        (1, 'a'),
        (2, 'b'),
        (3, 'c'),
    ]
    assert rich_helpers.to_string(rich_helpers.rows_to_table(rows, columns=('col1', 'col2'))) == """\
┏━━━━━━┳━━━━━━┓
┃ col1 ┃ col2 ┃
┡━━━━━━╇━━━━━━┩
│ 1    │ a    │
│ 2    │ b    │
│ 3    │ c    │
└──────┴──────┘
"""
