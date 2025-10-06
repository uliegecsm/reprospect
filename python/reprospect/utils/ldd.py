import pathlib
import re
import subprocess

import typeguard

@typeguard.typechecked
def get_shared_dependencies(*, file : pathlib.Path) -> set[pathlib.Path]:
    """
    Get the set of shared object dependencies.
    """
    libs = set()
    for lib in subprocess.check_output(['ldd', file]).decode().splitlines():
        match = re.search(r'[A-Za-z0-9.\+_\-]+ => ([A-Za-z0-9.\+_\-/]+) \(0x[0-9a-f]+\)', lib)
        if match is not None:
            libs.add(pathlib.Path(match.group(1)))
    return libs
