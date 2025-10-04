import argparse
import os
import pathlib
import sys
import typing
import unittest

import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import CuObjDump
from reprospect.tools.ncu          import Metric, MetricCorrelation, Report, Session
from reprospect.tools.sass         import Decoder

class Case(unittest.TestCase):
    """
    Base class for running a `ReProspect` case.
    """
    @classmethod
    def setUpClass(cls):
        cls._args, _ = cls._parse_args()

        # Environment that may be modified for running the case.
        cls._environ = os.environ.copy()

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
    
    @classmethod
    @typeguard.typechecked
    def output_path(cls) -> pathlib.Path:
        """
        Output path for the case.
        """
        output_path = pathlib.Path(str(cls._args.file) + "." + cls._args.name)
        output_path.mkdir(parents = True, exist_ok = True)
        return output_path

    @classmethod
    @typeguard.typechecked
    def cwd(cls) -> pathlib.Path:
        """
        Working directory for the case.
        """
        return cls._args.cwd

    @classmethod
    @typeguard.typechecked
    def cuobjdump(cls, *, arch : typing.Optional[NVIDIAArch] = None, sass : bool = True) -> CuObjDump:
        """
        Return a `CuObjDump` instance for the case file.
        """
        # todo: should there be an option to do the decode as well?
        if not arch:
            arch = NVIDIAArch.from_str('BLACKWELL120')
            # todo: deduce from compile commands

        cubin_file = cls.output_path() / f'{cls._args.file.name}.2.{arch.as_sm}.cubin'
        # todo: should we check in the binaries whether the cubin is in the file?

        cubin, _ = CuObjDump.extract(
            file  = cls._args.file,
            arch  = arch,
            cwd   = cls.cwd(),
            cubin = cubin_file.name,
            sass  = sass,
        )

        return cubin

    @classmethod
    @typeguard.typechecked
    def cuobjdump_and_decode(cls, *, arch : typing.Optional[NVIDIAArch] = None) -> (CuObjDump, Decoder):
        """
        Return a `CuObjDump` instance for the case file and decode the `SASS`.
        """
        cubin = cls.cuobjdump(arch = arch, sass = True)
        return (cubin, Decoder(code = cubin.sass))

    @classmethod
    @typeguard.typechecked
    def ncu(
        cls,
        *, 
        nvtx_capture : typing.Optional[str] = None,
        metrics      : typing.Optional[list[Metric | MetricCorrelation]] = None,
    ) -> Report:
        session = Session(output = cls.output_path() / cls._args.name)
        session.run(
            cmd = [cls._args.file],
            nvtx_capture = nvtx_capture,
            cwd = cls.cwd(),
            metrics = metrics,
        )
        return Report(path = cls.output_path(), name = cls._args.name)
