import pathlib
import re
import subprocess

import typeguard

@typeguard.typechecked
def get_shared_dependencies(*, file : str | pathlib.Path) -> list[pathlib.Path]:
    """
    Get the list of shared object dependencies.

    It assumes that `ldd` output follows::

        libname => /path/to/lib.so (address)
    """
    libs : list[pathlib.Path] = []
    for lib in subprocess.check_output(['ldd', file]).decode().splitlines():
        if (match := re.search(r'[A-Za-z0-9.\+_\-]+ => ([A-Za-z0-9.\+_\-/]+) \(0x[0-9a-f]+\)', lib)) is not None:
            libs.append(pathlib.Path(match.group(1)))
    return libs
