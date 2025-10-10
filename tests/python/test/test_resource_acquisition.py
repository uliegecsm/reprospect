import logging
import pathlib
import tempfile
import threading
import time

import pytest
import typeguard

from reprospect.test import resource_acquisition

class TestLocker:
    """
    Test :py:class:`reprospect.test.resource_acquisition.Locker`.
    """
    def test_acquire_single(self) -> None:
        """
        Acquire the resource when there is no contention for the resource.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with resource_acquisition.Locker(lock_dir = tmpdir) as locker:
                locker.acquire(resource = 'easy')
                locker.release(resource = 'easy')

    def test_acquire_many(self) -> None:
        """
        Acquire the resource when there is high contention for the resource.
        Ensures that resource acquisition is atomic across multiple threads.
        """
        NTHREADS = 5
        NITERATIONS = 5
        RESOURCE = 'shared'
        TIMEOUT = 60
        POLLING = 0.1
        TTL = 3600

        # Track violations of the atomicity.
        violations     = []
        lock_holder    = {'thread': None, 'count': 0}
        threading_lock = threading.Lock()

        @typeguard.typechecked
        def worker(*, file : pathlib.Path, lock_dir : str | pathlib.Path, index: int) -> None:
            for iteration in range(NITERATIONS):
                with resource_acquisition.Locker(lock_dir = lock_dir) as locker:
                    locker.acquire(resource = RESOURCE, ttl = TTL, timeout = TIMEOUT, polling = POLLING)

                    # Verify exclusive access.
                    with threading_lock:
                        if lock_holder['thread'] is not None:
                            violations.append(f"Thread {index} acquired lock while thread {lock_holder['thread']} still holds it.")
                        lock_holder['thread'] = index
                        lock_holder['count'] += 1

                    counter = int(file.read_text().strip())
                    counter += 1
                    file.write_text(str(counter))

                    # Small delay to increase chance of race conditions if locking fails.
                    time.sleep(0.001)

                    # Read back to verify.
                    readback = int(file.read_text().strip())
                    if readback != counter:
                        violations.append(f"Thread {index} wrote {counter} but read back {readback}.")

                    with threading_lock:
                        lock_holder['thread'] = None

                    locker.release(resource = RESOURCE)

                    # Small delay between iterations to vary contention patterns.
                    if iteration < NITERATIONS - 1:
                        time.sleep(0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            file = pathlib.Path(tmpdir) / 'counter.log'
            file.write_text(str(0))

            threads = [
                threading.Thread(target = worker, kwargs = {'file' : file, 'lock_dir' : tmpdir, 'index' : ithread})
                for ithread in range(NTHREADS)
            ]

            start = time.time()

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            end = time.time()

            logging.info(f'Elapsed time for {NTHREADS} threads acquiring {NITERATIONS} times the resource {RESOURCE} is {end - start} seconds.')
            logging.info(f'Average is {(end - start) / (NTHREADS * NITERATIONS)} seconds per acquisition.')
            logging.info(f'Total lock acquisitions tracked: {lock_holder["count"]}.')

            assert not violations

            assert int(file.read_text().strip()) == NTHREADS * NITERATIONS
            assert lock_holder['count']          == NTHREADS * NITERATIONS

    def test_release_without_acquire(self) -> None:
        """
        Check what happens when trying to release a resource that was not acquired.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with resource_acquisition.Locker(lock_dir = tmpdir) as locker:
                with pytest.raises(RuntimeError, match = 'The resource easy was not found.'):
                    locker.release(resource = 'easy')
