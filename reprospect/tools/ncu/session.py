import dataclasses
import logging
import os
import pathlib
import subprocess
import time
import typing

import attrs

from reprospect.tools.ncu.metrics import MetricKind, gather


@attrs.define(frozen=True, slots=True, kw_only=True)
class Command: # pylint: disable=too-many-instance-attributes
    """
    Run a ``ncu`` command line.
    """
    executable: str | pathlib.Path
    """Executable to run."""
    output: pathlib.Path
    """Report file."""
    opts: tuple[str, ...] = ()
    """Options that do not involve paths."""
    log: pathlib.Path = attrs.field(init=False)
    """Log file."""
    metrics: tuple[MetricKind, ...] | None = None
    """Metrics."""
    nvtx_includes: tuple[str, ...] | None = None
    """
    NVTX include.
    Refer to https://docs.nvidia.com/nsight-compute/2023.3/NsightComputeCli/index.html#nvtx-filtering.
    """
    args: tuple[str | pathlib.Path, ...] | None = None
    """Arguments to pass to the executable."""
    env: typing.Mapping[str, str] | None = None
    """Mapping used to update the environment before running, see :py:meth:`run`."""

    cmd: tuple[str | pathlib.Path, ...] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        """
        Enrich :py:attr:`opts` and build :py:attr:`cmd`.
        """
        # The log file is always next to the output.
        object.__setattr__(self, 'log', self.output.with_suffix('.log'))

        # Enrich the options.
        opts: tuple[str, ...] = self.opts + (
            '--print-summary=per-kernel',
            '--warp-sampling-interval=0',
        )
        if self.nvtx_includes:
            opts += (
                '--nvtx',
                '--print-nvtx-rename=kernel',
                *(f'--nvtx-include={x}' for x in self.nvtx_includes),
            )

        if self.metrics:
            opts += (f'--metrics={",".join(gather(metrics = self.metrics))}',)

        object.__setattr__(self, 'opts', opts)

        # Build the final full command.
        object.__setattr__(self, 'cmd', (
            'ncu',
            *self.opts,
            '--force-overwrite', '-o', self.output,
            '--log-file', self.log,
            self.executable,
            *(self.args or ()),
        ))

    def run(self, *,
        cwd: pathlib.Path | None = None,
        env: typing.MutableMapping[str, str] | None = None,
    ) -> int:
        if self.env is not None:
            if env is None:
                env = os.environ.copy()
            env.update(self.env)
        return subprocess.check_call(args=self.cmd, env=env, cwd=cwd)

@dataclasses.dataclass(frozen=True, slots=True)
class Session:
    """
    `Nsight Compute` session interface.
    """
    command: Command

    def run(
        self,
        cwd: pathlib.Path | None = None,
        env: typing.MutableMapping | None = None,
        retries: int = 1,
        sleep: typing.Callable[[int, int], float] = lambda retry, retries: 3. * (1. - retry / retries),
    ) -> None:
        """
        Run ``ncu`` using :py:attr:`command`.

        :param retries: ``ncu`` might fail acquiring some resources because other instances are running. Retry a few times.
                        See https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq (Profiling failed because a driver resource was unavailable).

        :param sleep: The time to sleep between successive retries.
                      The callable is given the current retry index (descending) and the amount of allowed retries.

        .. warning::

            According to https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#ElevPrivsTag,
            `GPU` performance counters are not available to all users by default.

        .. note::

            As of ``ncu`` `2025.1.1.0`, a note tells us that specified NVTX include expressions match only start/end ranges.

        References:

        * https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering
        """
        for retry in reversed(range(retries)):
            try:
                logging.info(f"Launching 'ncu' with {self.command.cmd} (log file at {self.command.log}).")
                self.command.run(cwd=cwd, env=env)
                break
            except subprocess.CalledProcessError:
                retry_allowed = False
                if retry > 0 and self.command.log.is_file():
                    with open(self.command.output.with_suffix('.log'), encoding='utf-8') as fin:
                        for line in fin:
                            if line.startswith('==ERROR== Profiling failed because a driver resource was unavailable.'):
                                logging.warning('Retrying because a driver resource was unavailable.')
                                retry_allowed = True
                                break

                if not retry_allowed:
                    logging.exception(
                        f"Failed launching 'ncu' with {self.command.cmd}."
                        "\n"
                        f"{self.command.log.read_text(encoding = 'utf-8') if self.command.log.is_file() else ''}",
                    )
                    raise
                sleep_for = sleep(retry, retries)
                logging.info(f'Sleeping {sleep_for} seconds before retrying.')
                time.sleep(sleep_for)
