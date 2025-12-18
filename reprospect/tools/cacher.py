import abc
import dataclasses
import datetime
import logging
import pathlib
import pickle
import sqlite3
import typing
import sys

import blake3

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

class Cacher(abc.ABC):
    """
    Cache results.

    The cacher stores the files as follows::

        ~/.<cache_dir>/
        ├── <hash>/
        │   ├── file_A
        │   ├── file_B
        │   └── ...
        └── cache.db

    Available hashes are stored in a `SQLITE` database (``cache.db``) that
    also stores the timestamp at which the entry was populated.

    Inspired by ``ccache``, see also https://ccache.dev/manual/4.12.1.html#_how_ccache_works.
    """
    #: Name of the table.
    TABLE: typing.ClassVar[str]

    @dataclasses.dataclass(frozen = True)
    class Entry:
        #: Set to `True` if the result was served from cache.
        cached: bool

        #: Digest of the entry.
        digest: str

        #: Timestamp at which the entry was populated.
        timestamp: datetime.datetime

        # Directory wherein files are stored.
        directory: pathlib.Path

    def __init__(self, directory: str | pathlib.Path) -> None:
        """
        :param directory: Where the cacher stores all files.
        """
        self.directory = pathlib.Path(directory)

        if not self.directory.is_dir():
            self.directory.mkdir(parents = True, exist_ok = True)

        self.file = self.directory / 'cache.db'

        self.database: sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        """
        Connect to the database.
        """
        self.database = self.create_db(file = self.file) if not self.file.is_file() else sqlite3.connect(self.file)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        assert self.database is not None
        self.database.close()

    def create_db(self, file: pathlib.Path) -> sqlite3.Connection:
        """
        Create the `SQLITE` database.
        """
        logging.info(f'Creating table \'{self.TABLE}\' in {self.file}.')
        db = sqlite3.connect(file)
        cursor = db.cursor()
        cursor.execute('\n'.join((
            f'CREATE TABLE IF NOT EXISTS {self.TABLE} (',
                'hash TEXT PRIMARY KEY,',
                'timestamp REAL',
            ')',
        )))
        db.commit()
        return db

    def hash(self, **kwargs) -> blake3.blake3:
        """
        This method is intended to be overridden in a child class with the specifics of your case.
        """
        hasher = blake3.blake3() # pylint: disable=not-callable

        for key, value in kwargs.items():
            hasher.update(pickle.dumps((key, value)))

        return hasher

    @abc.abstractmethod
    def populate(self, directory: pathlib.Path, **kwargs) -> None:
        pass

    def get(self, **kwargs) -> 'Cacher.Entry':
        """
        Serve cached entry on cache hit, populate on cache miss.
        """
        hasher = self.hash(**kwargs)

        hexdigest = hasher.hexdigest()

        assert self.database is not None

        with self.database:
            cursor = self.database.cursor()

            cursor.execute(f'SELECT timestamp FROM {self.TABLE} WHERE hash=?', (hexdigest,))
            row = cursor.fetchone()

            directory = self.directory / hexdigest

            directory.mkdir(parents = True, exist_ok = True)

            if row is None:
                logging.info(f'Cache miss (with hash {hexdigest}).')

                self.populate(directory = directory, **kwargs)

                entry = Cacher.Entry(cached = False, digest = hexdigest, timestamp = datetime.datetime.now(datetime.timezone.utc), directory = directory)

                cursor.execute(
                    f'REPLACE INTO {self.TABLE} (hash, timestamp) VALUES (?, ?)',
                    (entry.digest, entry.timestamp.isoformat()),
                )
            else:
                timestamp, = row

                logging.info(f'Cache hit dating from {timestamp} (with hash {hexdigest}).')

                entry = Cacher.Entry(cached = True, digest = hexdigest, timestamp = datetime.datetime.fromisoformat(timestamp), directory = directory)

            return entry
