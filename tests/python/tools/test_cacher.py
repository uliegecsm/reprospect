import pathlib
import tempfile

from reprospect.tools.cacher import Cacher

class TestCacher:
    """
    Tests for :py:class:`reprospect.tools.cacher.Cacher`.
    """
    def test_hash_same(self):
        """
        Check that calling :py:meth:`reprospect.tools.cacher.Cacher.hash` twice with the same values returns the same hash.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir) as cacher:
                hash_a = cacher.hash(opts = ['--nvtx'], exe = 'my-exec', args = ['--bla=42'])
                hash_b = cacher.hash(opts = ['--nvtx'], exe = 'my-exec', args = ['--bla=42'])

                assert hash_a.digest() == hash_b.digest()

    def test_hash_different(self):
        """
        Check that calling :py:meth:`reprospect.tools.cacher.Cacher.hash` twice with different values returns different hashes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir) as cacher:
                # Different 'opts'.
                hash_a = cacher.hash(opts = ['--ntvx'], exe = pathlib.Path('my-exec'))
                hash_b = cacher.hash(opts = ['--nvtx'], exe = pathlib.Path('my-exec'))

                assert hash_a.digest() != hash_b.digest()

                # Different 'args'.
                hash_a = cacher.hash(opts = ['--nvtx'], exe = pathlib.Path('my-exec'), args = ['1'])
                hash_b = cacher.hash(opts = ['--nvtx'], exe = pathlib.Path('my-exec'), args = ['2'])

                assert hash_a.digest() != hash_b.digest()

                # Different 'env'.
                hash_a = cacher.hash(opts = ['--nvtx'], exe = pathlib.Path('my-exec'), args = ['42'], env = {'my' : 'env-1'})
                hash_b = cacher.hash(opts = ['--nvtx'], exe = pathlib.Path('my-exec'), args = ['42'], env = {'my' : 'env-2'})

                assert hash_a.digest() != hash_b.digest()

    def test_cache_miss(self):
        """
        Check that calling :py:meth:`reprospect.tools.cacher.Cacher.get` for the first time leads to a cache miss.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir) as cacher:
                results = cacher.get(opts = ['--nvtx'], exe = 'test-exec', args = ['--bla=42'])

                assert results.cached == False

    def test_cache_hit(self):
        """
        Check that calling :py:meth:`reprospect.tools.cacher.Cacher.get` for the second time leads to a cache hit.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cacher(directory = tmpdir) as cacher:
                results_first = cacher.get(opts = ['--nvtx'], exe = 'exec-666', args = ['--bla=42'], env = {'my' : 'env'}, anything_you_want = {4: 2})

                assert results_first.cached == False

                results_second = cacher.get(opts = ['--nvtx'], exe = 'exec-666', args = ['--bla=42'], env = {'my' : 'env'}, anything_you_want = {4: 2})

                assert results_second.cached == True
                assert results_second.digest == results_first.digest
                assert results_second.timestamp == results_first.timestamp
                assert results_second.directory == results_first.directory
