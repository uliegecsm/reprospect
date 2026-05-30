import pathlib
import typing

import pytest

from reprospect.tools.binaries.symtab import get_symbol_table
from reprospect.utils import cmake

from tests.compilation import get_compilation_output
from tests.parameters import PARAMETERS, Parameters


@pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
class TestGetSymbolTable:
    """
    Tests for :py:func:`reprospect.tools.binaries.symtab.get_symbol_table`.
    """
    CPP_FILE: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'assets' / 'many.cpp'

    FUNCTIONS: typing.Final[tuple[str, ...]] = (
        'say_hi',
        'vector_atomic_add_42',
    )

    def test(self, workdir: pathlib.Path, parameters: Parameters, cmake_file_api: cmake.FileAPI) -> None:
        """
        Compile :py:attr:`CPP_FILE` as an executable, load the symbol table.
        The symbol table must contain symbols that match :py:attr:`FUNCTIONS`.
        """
        output, _ = get_compilation_output(
            source=self.CPP_FILE,
            cwd=workdir,
            arch=parameters.arch,
            object_file=False,
            cmake_file_api=cmake_file_api,
        )

        st = get_symbol_table(file=output)

        assert all(st['name'].str.contains(x, regex=False).any() for x in self.FUNCTIONS)
