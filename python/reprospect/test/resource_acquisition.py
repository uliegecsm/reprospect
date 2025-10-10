import argparse
import datetime
import json
import logging
import os
import pathlib
import sqlite3
import time
import typing

import typeguard

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--resource', type = json.loads, required = True, help = 'JSON-like description of the resource.')
    parser.add_argument('--action', choices = ['acquire', 'release', 'status', 'cleanup'], required = True)
    parser.add_argument('--timeout', type = float, default = 60., help= 'Timeout in seconds.')
    parser.add_argument('--ttl', type = int, default = 3600, help = 'Lock time-to-live (TTL) in seconds.')

    return parser.parse_args()

class Transaction:
    """
    Use `BEGIN IMMEDIATE` to acquire a reserved lock immediately,
    preventing other writers from starting transactions until this one commits.
    This guarantees that the insert is atomic relative to other insert attempts.
    """
    @typeguard.typechecked
    def __init__(self, connection : sqlite3.Connection) -> None:
        self.connection = connection

    @typeguard.typechecked
    def __enter__(self) -> sqlite3.Cursor:
        return self.connection.execute("BEGIN IMMEDIATE")

    @typeguard.typechecked
    def __exit__(self, exc_type : typing.Optional[typing.Any], *args, **kwargs) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()

class Locker:
    """
    Use a `SQLITE` database to atomically acquire/release resources by name.
    """
    @typeguard.typechecked
    def __init__(self, lock_dir : typing.Optional[str | pathlib.Path] = None) -> None:
        self.lock_dir = pathlib.Path(lock_dir if lock_dir is not None else os.environ.get("RESOURCE_LOCK_DIR", "/resource-locks"))

        self.lock_dir.mkdir(parents = True, exist_ok = True)

        self.lock_db = self.lock_dir / 'locks.db'

        logging.info(f'Using lock database at {self.lock_db}.')

    @typeguard.typechecked
    def __enter__(self) -> 'Locker':
        """
        Connect to the lock database.
        """
        self.connection = self.connect()
        return self

    @typeguard.typechecked
    def __exit__(self, *args, **kwargs) -> None:
        """
        Disconnect.
        """
        self.connection.close()

    @typeguard.typechecked
    def to_key(self, resource : typing.Any) -> str:
        """
        Convert `resource` to a `JSON`-based string key.
        """
        return json.dumps(resource, sort_keys = True, separators = (',', ':'))

    @typeguard.typechecked
    def connect(self) -> sqlite3.Connection:
        """
        Connect to the database, create the table if needed.

        Each entry is a resource name, and has the timestamp at which it was acquired.
        """
        connection = sqlite3.connect(self.lock_db, isolation_level = 'DEFERRED')

        # If creating the table fails, it means the table is probably already setup correctly.
        try:
            connection.execute("""
CREATE TABLE resource_locks (
    id INTEGER PRIMARY KEY,
    resource_json TEXT NOT NULL UNIQUE,
    pid INTEGER NOT NULL,
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    hostname TEXT
)
""")
            # Enable WAL mode for better concurrency.
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass

        return connection

    @typeguard.typechecked
    def acquire(self, resource : typing.Any, timeout : float = 60., polling : float = 1., ttl : int = 60) -> None:
        """
        Acquire `resource`. Blocks until acquired or timeout.

        :param timeout: Time (in seconds) before giving up.
        :param polling: Time (in seconds) before retrying.
        """
        key = self.to_key(resource = resource)
        nodename = os.uname().nodename
        pid = os.getpid()

        start = time.time()

        while time.time() - start < timeout:
            try:
                with Transaction(connection = self.connection):
                    now = datetime.datetime.now()
                    expires_at = (now + datetime.timedelta(seconds = ttl)).isoformat()

                    cursor = self.connection.execute(
"""
INSERT INTO resource_locks (resource_json, pid, expires_at, hostname)
VALUES (?, ?, ?, ?)
RETURNING acquired_at
""",
                        (key, pid, expires_at, nodename),
                    )

                    logging.info(f'Resource {resource} acquired at {cursor.fetchone()[0]}.')
                    return

            except sqlite3.IntegrityError as e:
                if 'UNIQUE constraint failed: resource_locks.resource_json' in str(e):
                    time.sleep(polling)
                else:
                    raise

        raise TimeoutError(f'Could not acquire resource {self.resource} within {timeout} second(s).')

    @typeguard.typechecked
    def release(self, resource : typing.Any) -> None:
        """
        Release the resource.
        """
        try:
            key = self.to_key(resource = resource)
            with Transaction(connection = self.connection):
                cursor = self.connection.execute(
                    "DELETE FROM resource_locks WHERE resource_json = ?", (key,)
                )

                if cursor.rowcount == 0:
                    raise RuntimeError(f'The resource {resource} was not found.')
                else:
                    logging.info(f'Resource {resource} released at {datetime.datetime.now()}.')

        except sqlite3.OperationalError as e:
            logging.exception(e)
            raise RuntimeError(f'Failed to release resource {resource}.')

    @typeguard.typechecked
    def status(self) -> None:
        """
        Current lock status.
        """
        with Transaction(connection = self.connection):
            locks = self.connection.execute(
                'SELECT * FROM resource_locks ORDER BY acquired_at'
            )

            if locks.rowcount <= 0:
                logging.info(f'No active lock in {self.lock_db}.')
            else:
                logging.info(f'Active locks:')
                for lock in locks:
                    logging.info(f'\t{lock}')

    @typeguard.typechecked
    def cleanup(self) -> None:
        """
        Remove all expired locks.
        """
        with Transaction(connection = self.connection):
            logging.info(f'Cleaning up expired locks in {self.lock_db}.')
            cursor = self.connection.execute(
                'DELETE FROM resource_locks WHERE expires_at IS NOT NULL AND expires_at < (?)',
                (datetime.datetime.now().isoformat(),),
            )
            logging.info(f'{cursor.rowcount} entries were deleted.')

def main() -> None:

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    with Locker() as locker:

        match args.action:
            case 'acquire':
                locker.acquire(resource = args.resource)
            case 'release':
                locker.release(resource = args.resource)
            case 'status':
                locker.status()
            case 'cleanup':
                locker.cleanup()
            case _:
                raise ValueError(f'unsupported action {args.action}')

if __name__ == "__main__":

    main()
