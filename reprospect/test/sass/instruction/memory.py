import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum.strenum import StrEnum

class MemorySpace(StrEnum):
    """
    Allowed memory spaces.
    """
    GENERIC = ''
    GLOBAL = 'G'
    LOCAL = 'L'
    SHARED = 'S'
