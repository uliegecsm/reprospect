import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import typing

import blake3

from reprospect.tools import cacher
from reprospect.tools.ncu.session import Command, Session
from reprospect.utils import ldd

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Cacher(cacher.Cacher):
    """
    Cacher tailored to ``ncu`` results.

    ``ncu`` require quite some time to acquire results, especially when there are many kernels to profile and/or
    many metrics to collect.

    On a cache hit, the cacher will serve:

    - ``<cache_key>.ncu-rep`` file
    - ``.log`` file

    On a cache miss, ``ncu`` is launched and the cache entry populated accordingly.

    .. note::

        It is assumed that hashing is faster than running ``ncu`` itself.

    .. warning::

        The cache should not be shared between machines, since there may be differences between machines
        that influence the results but are not included in the hashing.
    """
    TABLE : typing.ClassVar[str] = 'ncu'

    def __init__(self, *, directory : typing.Optional[str | pathlib.Path] = None):
        super().__init__(directory = directory or (pathlib.Path(os.environ['HOME']) / '.ncu-cache'))

    def hash_impl(self, *, command : Command) -> blake3.blake3:
        """
        Hash based on:

        * ``ncu`` version
        * ``ncu`` options (but not the output and log files)
        * executable content
        * executable arguments
        * linked libraries
        * environment
        """
        hasher = blake3.blake3() # pylint: disable=not-callable

        hasher.update(subprocess.check_output(('ncu', '--version')))

        if command.opts:
            hasher.update(shlex.join(command.opts).encode())

        hasher.update_mmap(command.executable)

        if command.args:
            hasher.update(shlex.join(map(str, command.args)).encode())

        if command.env:
            hasher.update(json.dumps(command.env).encode())

        for lib in sorted(ldd.get_shared_dependencies(file = command.executable)):
            hasher.update_mmap(lib)

        return hasher

    @override
    def hash(self, **kwargs) -> blake3.blake3:
        return self.hash_impl(command = kwargs['command'])

    @override
    def populate(self, directory : pathlib.Path, **kwargs) -> None:
        """
        When there is a cache miss, call :py:meth:`reprospect.tools.ncu.Session.run`.
        Fill the `directory` with the artifacts.
        """
        command = kwargs.pop('command')

        Session(command = command).run(**kwargs)

        shutil.copy(dst = directory, src = command.output.with_suffix('.ncu-rep'))
        shutil.copy(dst = directory, src = command.log)

    def run(self,
        command : Command,
        **kwargs,
    ) -> cacher.Cacher.Entry:
        """
        On a cache hit, copy files from the cache entry.
        """
        entry = self.get(command = command, **kwargs)

        if entry.cached:
            shutil.copytree(entry.directory, command.output.parent, dirs_exist_ok = True)

        return entry
