import pathlib

import pandas

from reprospect.tools.binaries.elf import ELF

def get_symbol_table(*, file : pathlib.Path) -> pandas.DataFrame:
    """
    Extract the symbol table from `file`.
    """
    with ELF(file = file) as elf:
        assert elf.elf is not None
        [section] = elf.elf.iter_sections(type = 'SHT_SYMTAB')

        return pandas.DataFrame(
            data = (
                (
                    idx,
                    symbol['st_value'],
                    symbol['st_size'],
                    symbol['st_info']['bind'],
                    symbol['st_info']['type'],
                    symbol['st_other']['visibility'],
                    symbol['st_shndx'],
                    symbol.name or '(null)',
                )
                for idx, symbol in enumerate(section.iter_symbols())
            ),
            columns = ('index', 'value', 'size', 'bind', 'type', 'visibility', 'shndx', 'name'),
        )
