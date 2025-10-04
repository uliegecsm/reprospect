import argparse
import functools
import os
import pathlib
import sys
import typing
import unittest

import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump

class Case(unittest.TestCase):
    """
    Base class for running a `ReProspect` case.
    """
    def setUp(self):
        self._args, _ = self._parse_args()

        # Environment that may be modified for running the case.
        self._environ = os.environ.copy()

    @staticmethod
    @typeguard.typechecked
    def _parse_args(sysargs : list[str] = sys.argv[1:]) -> (argparse.Namespace, list[str]):
        """
        Parse CLI arguments.
        """
        parser = argparse.ArgumentParser(allow_abbrev = False)

        parser.add_argument('--name',             help = "Name of the case.",                type = str,          required = True,)
        parser.add_argument('--file',             help = "File.",                            type = pathlib.Path, required = True,)
        parser.add_argument('--compile-commands', help = "Compile commands JSON file path.", type = pathlib.Path, required = False,)
        parser.add_argument('--cwd',              help = "Working directory.",               type = pathlib.Path, required = False,)

        return parser.parse_known_args(args = sysargs)
    
    @property
    @typeguard.typechecked
    def output_path(self) -> pathlib.Path:
        """
        Output path for the case.
        """
        output_path = pathlib.Path(str(self._args.file) + "." + self._args.name)
        output_path.mkdir(parents = True, exist_ok = True)
        return output_path

    @property
    @typeguard.typechecked
    def cwd(self) -> pathlib.Path:
        """
        Working directory for the case.
        """
        return self._args.cwd
    
    @functools.cached_property
    @typeguard.typechecked
    def cuobjdump(self, *, arch : typing.Optional[NVIDIAArch] = None, sass : bool = True) -> CuObjDump:
        """
        Return a `CuObjDump` instance for the case file.
        """
        # todo: should there be an option to do the decode as well?
        if not arch:
            arch = NVIDIAArch.from_str('BLACKWELL120')
            # todo: deduce from compile commands

        cubin_file = self.output_path / f'{self._args.file.name}.2.{arch.as_sm}.cubin'
        # todo: should we check in the binaries whether the cubin is in the file?

        cubin, _ = CuObjDump.extract(
            file = self._args.file,
            arch   = arch,
            cwd    = self.cwd,
            cubin  = cubin_file.name,
            sass   = sass,
        )

        return cubin
