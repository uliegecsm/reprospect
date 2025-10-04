import sys
import unittest

from .case import Case

def main():
    _, remaining = Case._parse_args(sys.argv[1:])
    unittest.main(argv = [sys.argv[0]] + remaining)
