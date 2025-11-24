import logging
import pathlib
import subprocess
import typing

def popen_stream(*,
    args : typing.Sequence[str | pathlib.Path],
    **kwargs,
) -> typing.Generator[str, None, None]:
    """
    Yield lines from a :py:class:`subprocess.Popen` lazily, with robust error handling and resource cleanup.

    :param args: Command to run.
    :param kwargs: Additional arguments to pass to :py:class:`subprocess.Popen`.

    1. Launch a subprocess with both `stdout` and `stderr` captured as text streams.
    2. Lazily yield each line from `stdout` as it becomes available.
    3. After `stdout` is exhausted, wait for the process to finish.
    4. If the process exits with a nonzero return code, raise
       :py:class:`subprocess.CalledProcessError` with captured `stderr`.
    5. Guarantee resource cleanup:

        a. If the process hasn't finished, it is terminated.
        b. If termination fails (within 2 seconds), the process is forcibly killed.
    """
    with subprocess.Popen(
        args = args,
        stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        text = True,
        **kwargs,
    ) as process:
        try:
            assert process.stdout is not None

            yield from process.stdout

            if (returncode := process.wait()) != 0:
                assert process.stderr is not None
                raise subprocess.CalledProcessError(
                    returncode = returncode,
                    cmd = args,
                    stderr = process.stderr.read(),
                )

        finally:
            if process.poll() is None:
                logging.warning(f'Terminating {process!r}.')
                process.terminate()
                try:
                    process.wait(timeout = 2)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    logging.warning(f'Killing process {process!r}.')
                    process.kill()
                    process.wait()
