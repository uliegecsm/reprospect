import pathlib
import re
import typing

from reprospect.utils.subprocess_helpers import popen_stream

def get_shared_dependencies(*, file: str | pathlib.Path) -> typing.Generator[pathlib.Path, None, None]:
    """
    Get the list of shared object dependencies.

    It assumes that ``ldd`` output follows::

        libname => /path/to/lib.so (address)
    """
    return (
        pathlib.Path(match.group(1))
        for lib in popen_stream(args = ('ldd', file))
        if (match := re.search(r'[A-Za-z0-9.\+_\-]+ => ([A-Za-z0-9.\+_\-/]+) \(0x[0-9a-f]+\)', lib)) is not None
    )
