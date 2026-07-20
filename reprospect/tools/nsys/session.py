# pylint: disable=duplicate-code
import dataclasses
import logging
import os
import pathlib
import subprocess
import typing

import attrs
import pandas


@attrs.define(frozen=True, slots=True, kw_only=True)
class Command: # pylint: disable=too-many-instance-attributes
    """
    Run a ``nsys`` command line.
    """
    executable: str | pathlib.Path
    """Executable to run."""
    output: pathlib.Path
    """Report file."""
    opts: tuple[str, ...] = ()
    """Options that do not involve paths."""
    nvtx_capture: str | None = None
    """NVTX capture string."""
    capture_range_end: str = 'stop'
    """NVTX capture range end."""
    args: tuple[str | pathlib.Path, ...] | None = None
    """Arguments to pass to the executable."""
    env: typing.Mapping[str, str] | None = None
    """Mapping used to update the environment before running, see :py:meth:`run`."""

    cmd: tuple[str | pathlib.Path, ...] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        """
        Enrich :py:attr:`opts` and build :py:attr:`cmd`.
        """
        if self.output.suffix != '.nsys-rep':
            object.__setattr__(self, 'output', self.output.parent / f'{self.output.name}.nsys-rep')

        # Disable collecting CPU samples.
        opts: tuple[str, ...] = self.opts + (
            '--sample=none',
            '--backtrace=none',
            '--cpuctxsw=none',
        )

        # We want to start data collection when the first NVTX range is met.
        # This reduces the amount of data collected (and makes things faster).
        if self.nvtx_capture is not None:
            match self.nvtx_capture:
                case '*':
                    pass
                case _:
                    opts += (
                        '--capture-range=nvtx',
                        f'--capture-range-end={self.capture_range_end}',
                        f'--nvtx-capture={self.nvtx_capture}',
                    )
            opts += ('--trace=nvtx,cuda',)
        else:
            opts += ('--trace=cuda',)

        object.__setattr__(self, 'opts', opts)

        # Build the final full command.
        object.__setattr__(self, 'cmd', (
            'nsys', 'profile',
            *self.opts,
            '--force-overwrite=true', '-o', self.output,
            self.executable,
            *(self.args or ()),
        ))

    def run(self, *,
        cwd: pathlib.Path | None = None,
        env: typing.MutableMapping[str, str] | None = None,
    ) -> int:
        if (self.nvtx_capture is not None or self.env is not None) and env is None:
            env = os.environ.copy()

        # For '--capture-range=nvtx' to accept our custom strings, we need to allow unregistered
        # strings to be considered.
        # See https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-interactive-cli-command-sequences.
        if self.nvtx_capture:
            assert env is not None
            env['NSYS_NVTX_PROFILER_REGISTER_ONLY'] = '0'

        if self.env:
            assert env is not None
            env.update(self.env)

        self.output.unlink(missing_ok=True)
        return subprocess.check_call(args=self.cmd, env=env, cwd=cwd)

@dataclasses.dataclass(frozen=True, slots=True)
class Session:
    """
    `Nsight Systems` session interface.
    """
    command: Command

    def run(
        self,
        cwd: pathlib.Path | None = None,
        env: typing.MutableMapping[str, str] | None = None,
    ) -> None:
        """
        Run ``nsys`` using :py:attr:`command`.
        """
        logging.info(f"Launching 'nsys' with {self.command.cmd}.")
        self.command.run(cwd=cwd, env=env)

    def export_to_sqlite(
        self,
        cwd: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """
        Export report to ``.sqlite``.
        """
        output_file_sqlite = self.command.output.with_suffix('.sqlite')

        cmd: tuple[str | pathlib.Path, ...] = (
            'nsys', 'export',
            '--type', 'sqlite',
            f'--output={output_file_sqlite}',
            self.command.output,
        )

        logging.info(f"Exporting to 'sqlite' with {cmd}.")
        output_file_sqlite.unlink(missing_ok=True)
        subprocess.check_call(cmd, cwd=cwd)

        return output_file_sqlite

    def extract_statistical_report(
        self,
        report: str = 'cuda_api_sum',
        filter_nvtx: str | None = None,
        cwd: pathlib.Path | None = None,
    ) -> pandas.DataFrame:
        """
        Extract `report`, filtering the database with `filter_nvtx`.
        """
        output_file_sqlite = self.command.output.with_suffix('.sqlite')

        cmd: tuple[str | pathlib.Path, ...] = (
            'nsys', 'stats',
            f'--output={self.command.output.parent / self.command.output.stem}',
            f'--report={report}',
            '--format=csv',
            '--timeunit=usec',
            *(('--filter-nvtx=' + filter_nvtx,) if filter_nvtx else ()),
            output_file_sqlite,
        )

        # 'nsys stats' will output to a file whose name follows the convention
        #    <basename>_<analysis&args>.<output_format>
        suffix = '_nvtx=' + filter_nvtx.replace('/', '-') if filter_nvtx else ''
        output_file_csv = self.command.output.parent / f'{self.command.output.stem}_{report}{suffix}.csv'

        logging.info(f'Extracting statistical report \'{report}\' from {output_file_sqlite} with {cmd}.')
        output_file_csv.unlink(missing_ok=True)
        subprocess.check_call(cmd, cwd=cwd)
        assert output_file_csv.is_file()

        return pandas.read_csv(output_file_csv)
