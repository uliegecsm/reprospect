import io
import logging
import pathlib
import re
import subprocess
import sys
import tempfile
import typing
import unittest.mock

import pytest

from reprospect.utils.subprocess_helpers import popen_stream


class TestPopenStream:
    """
    Tests for :py:class:`reprospect.utils.subprocess_helpers.popen_stream`.
    """
    @staticmethod
    def mocker(*, wait: int, stdout: str | None = None, stderr: str | None = None) -> unittest.mock.MagicMock:
        mocker = unittest.mock.MagicMock()

        mocker.stdout = io.StringIO(stdout)
        mocker.stderr = io.StringIO(stderr)
        mocker.wait = unittest.mock.Mock(return_value=wait)

        mocker.__enter__ = unittest.mock.Mock(return_value=mocker)
        mocker.__exit__ = unittest.mock.Mock(return_value=None)

        return mocker

    def test_str(self) -> None:
        """
        The command is composed of strings.
        """
        with unittest.mock.patch('subprocess.Popen', return_value=self.mocker(stdout='line1\nline2\nline3\n', wait=0)) as process:
            result = tuple(popen_stream(args=('echo', 'hello')))

            assert result == ('line1\n', 'line2\n', 'line3\n')

            process.assert_called_once_with(
                args=('echo', 'hello'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_path(self) -> None:
        """
        The command contains :py:class:`pathlib.Path`.
        """
        with unittest.mock.patch('subprocess.Popen', return_value=self.mocker(stdout='hello', wait=0)) as process:
            result = tuple(popen_stream(args=(pathlib.Path('echo'), 'hello')))

            assert result == ('hello',)

            process.assert_called_once_with(
                args=(pathlib.Path('echo'), 'hello'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_lazy(self) -> None:
        """
        Use an iterator to read line-by-line.
        """
        with unittest.mock.patch('subprocess.Popen', return_value=self.mocker(stdout='a\nb\nc', wait=0)):
            gen = popen_stream(args=('echo', 'hello'))
            first = next(gen)

            assert first == 'a\n'

            remaining = tuple(gen)

            assert remaining == ('b\n', 'c')

    def test_command_failure(self) -> None:
        """
        The command fails.
        """
        with unittest.mock.patch('subprocess.Popen', return_value=self.mocker(wait=42)):
            with pytest.raises(subprocess.CalledProcessError, match=r"Command '\('false',\)' returned non-zero exit status 42."):
                next(popen_stream(args=('false',)))

    def test_command_not_found(self) -> None:
        """
        The command is not found.
        """
        with unittest.mock.patch('subprocess.Popen', side_effect=FileNotFoundError("[Errno 2] No such file or directory: 'nonexistent_command_12345'")):
            with pytest.raises(FileNotFoundError, match='nonexistent_command_12345'):
                next(popen_stream(args=('nonexistent_command_12345',)))

    def test_empty_output(self) -> None:
        """
        Empty `stdout`.
        """
        with unittest.mock.patch('subprocess.Popen', return_value=self.mocker(wait=0)):
            result = tuple(popen_stream(args=('true',)))
            assert not result

    def test_generator_break(self, caplog) -> None:
        """
        Break while the generator is not finished.
        """
        LINES: typing.Final[tuple[str, ...]] = ('one', 'two', 'three', 'four', 'five')

        CODE: typing.Final[str] = f"""\
LINES = {LINES}
for line in LINES:
    print(line)
"""

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py') as tmpfile:

            tmpfile.write(CODE)
            tmpfile.flush()

            count = 0

            with caplog.at_level(logging.WARNING):
                for line in popen_stream(args=(sys.executable, tmpfile.name)):
                    if line.startswith('three'):
                        assert count == 2
                        break
                    assert line.rstrip('\n') in LINES
                    assert count < 2
                    count += 1

                assert count == 2 and len(LINES) == 5

            assert re.search(pattern=r'WARNING\s+root:subprocess_helpers.py:[0-9]+\s+Terminating <Pope', string=caplog.text) is not None, caplog.text
