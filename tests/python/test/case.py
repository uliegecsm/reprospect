import json
import logging
import pathlib

import pytest
import typeguard

from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.binaries     import get_arch_from_compile_command

class Case:
    """
    Base class for carrying out analyses of a given target file.
    """
    @pytest.fixture(scope = 'class')
    @classmethod
    @typeguard.typechecked
    def target_file(cls, request) -> pathlib.Path:
        return pathlib.Path(request.config.getoption('--target-file'))

    @pytest.fixture(scope = 'class')
    @classmethod
    @typeguard.typechecked
    def cwd(cls, target_file : pathlib.Path) -> pathlib.Path:
        """
        The working directory for the analysis.
        """
        cwd = target_file.parent / (target_file.name + '-case')
        cwd.mkdir(parents = False, exist_ok = True)
        return cwd

    @pytest.fixture(scope = 'class')
    @classmethod
    def arch(cls, request) -> NVIDIAArch:
        """
        Retrieve the GPU architecture from the compile command.

        We assume the target file was compiled for only a single architecture.
        """
        with open(request.config.getoption('--compile-commands')) as fin:
            commands = json.load(fin)
        logging.info(f'Looking for {cls.TARGET_SOURCE} in {commands}')
        archs = get_arch_from_compile_command(cmd = next(filter(
            lambda x: str(cls.TARGET_SOURCE) in x['file'],
            commands))['command']
        )
        assert len(archs) == 1
        return archs.pop()
