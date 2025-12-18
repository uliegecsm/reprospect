import re
import subprocess

import semantic_version


def get_version() -> semantic_version.Version:
    """
    Get version of ``nvcc``.
    """
    output = subprocess.check_output(('nvcc', '--version')).decode()

    if (matched := re.search(pattern = r'release ([0-9]+).([0-9]+), V([0-9]+).([0-9]+).([0-9]+)', string = output)) is None:
        raise RuntimeError('nvcc --version cannot be parsed.')

    assert matched.group(1) == matched.group(3)
    assert matched.group(2) == matched.group(4)

    return semantic_version.Version(major = int(matched.group(3)), minor = int(matched.group(4)), patch = int(matched.group(5)))
