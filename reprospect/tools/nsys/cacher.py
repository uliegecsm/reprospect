# pylint: disable=duplicate-code
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import typing

import blake3

from reprospect.tools import cacher
from reprospect.tools.nsys.session import Command, Session
from reprospect.utils import ldd

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Cacher(cacher.Cacher):
    """
    Cacher tailored to ``nsys`` results.

    ``nsys`` runs require quite some time to acquire results.

    On a cache hit, the cacher will serve:

    - ``.nsys-rep`` file
    - ``.sqlite`` file

    On a cache miss, ``nsys`` is launched and the cache entry populated accordingly.

    .. note::

        It is assumed that hashing is faster than running ``nsys`` itself.

    .. warning::

        The cache should not be shared between machines, since there may be differences between machines
        that influence the results but are not included in the hashing.
    """
    TABLE: typing.ClassVar[str] = 'nsys'

    def __init__(self, *, directory: str | pathlib.Path | None = None):
        super().__init__(directory=directory or (pathlib.Path(os.environ['HOME']) / '.nsys-cache'))

    def hash_impl(self, *, command: Command) -> blake3.blake3:
        """
        Hash based on:

        * ``nsys`` version
        * ``nsys`` options (but not the output files)
        * executable content
        * executable arguments
        * linked libraries
        * environment
        """
        hasher = blake3.blake3() # pylint: disable=not-callable

        hasher.update(subprocess.check_output(('nsys', '--version')))

        if command.opts:
            hasher.update(shlex.join(command.opts).encode())

        hasher.update_mmap(command.executable)

        if command.args:
            hasher.update(shlex.join(map(str, command.args)).encode())

        if command.env:
            hasher.update(json.dumps(command.env).encode())

        for lib in sorted(ldd.get_shared_dependencies(file=command.executable)):
            hasher.update_mmap(lib)

        return hasher

    @override
    def hash(self, **kwargs) -> blake3.blake3:
        return self.hash_impl(command=kwargs['command'])

    @override
    def populate(self, directory: pathlib.Path, **kwargs) -> None:
        """
        When there is a cache miss, call :py:meth:`reprospect.tools.nsys.Session.run`.
        Fill the `directory` with the artifacts.
        """
        command = kwargs.pop('command')

        Session(command=command).run(**kwargs)

        shutil.copy(dst=directory, src=command.output)

    def run(self, command: Command, **kwargs) -> cacher.Cacher.Entry:
        """
        On a cache hit, copy files from the cache entry.
        """
        entry = self.get(command=command, **kwargs)

        if entry.cached:
            shutil.copytree(entry.directory, command.output.parent, dirs_exist_ok=True)

        return entry

    @staticmethod
    def export_to_sqlite(
        command: Command,
        entry: cacher.Cacher.Entry,
        **kwargs,
    ) -> pathlib.Path:
        """
        Export report to ``.sqlite``.
        """
        output_file_sqlite = command.output.with_suffix('.sqlite')

        cached = entry.directory / output_file_sqlite.name

        if cached.is_file():
            logging.info(f'Serving {output_file_sqlite} from the cache entry {entry}.')
            shutil.copyfile(src=cached, dst=output_file_sqlite)
        else:
            logging.info(f'Populating the cache entry {entry} with {output_file_sqlite} from the cache entry {entry}.')
            Session(command=command).export_to_sqlite(**kwargs)
            shutil.copyfile(src=output_file_sqlite, dst=cached)

        return output_file_sqlite
