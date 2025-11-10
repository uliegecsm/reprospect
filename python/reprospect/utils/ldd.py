import pathlib
import re
import subprocess

def get_shared_dependencies(*, file : str | pathlib.Path) -> list[pathlib.Path]:
    """
    Get the list of shared object dependencies.

    It assumes that ``ldd`` output follows::

        libname => /path/to/lib.so (address)
    """
    lines = subprocess.check_output(('ldd', file)).decode().splitlines()
    return [
        pathlib.Path(match.group(1))
        for lib in lines
        if (match := re.search(r'[A-Za-z0-9.\+_\-]+ => ([A-Za-z0-9.\+_\-/]+) \(0x[0-9a-f]+\)', lib)) is not None
    ]
