import argparse
import io
import logging
import shutil
import subprocess
import typing

import numpy
import pandas

from reprospect.tools import architecture


class GPUDetector:
    """
    Detect all available GPUs using `nvidia-smi`.

    .. note::

        By default, results are cached.
    """
    COLUMNS: typing.ClassVar[dict[str, type[str] | numpy.dtype]] = {'uuid': str, 'index': numpy.dtype('int32'), 'name': str, 'compute_cap': str}

    _cache: typing.ClassVar[pandas.DataFrame | None] = None

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the cached `GPU` detection results.
        """
        cls._cache = None

    @classmethod
    def get(cls, *, cache: bool = True, enrich: bool = True) -> pandas.DataFrame:
        if cache and cls._cache is not None:
            return cls._cache

        results = cls.detect(enrich = enrich)

        if cache:
            cls._cache = results

        return results

    @classmethod
    def detect(cls, *, enrich: bool = True) -> pandas.DataFrame:
        """
        Implementation of the detection.
        """
        CMD: tuple[str, ...] = ('nvidia-smi', '--query-gpu=' + ','.join(cls.COLUMNS.keys()), '--format=csv') # pylint: disable=invalid-name

        if shutil.which('nvidia-smi') is not None:
            gpus = pandas.read_csv(
                io.StringIO(subprocess.check_output(CMD).decode()),
                sep = ',',
                skipinitialspace = True,
                dtype = cls.COLUMNS, # type: ignore[arg-type]
            )
            if not set(cls.COLUMNS.keys()).issubset(gpus.columns):
                raise RuntimeError(gpus.columns)
            if enrich:
                gpus.loc[:, 'architecture'] = gpus['compute_cap'].apply(
                    lambda x: architecture.NVIDIAArch.from_compute_capability(int(x.replace('.', ''))),
                )
            return gpus
        logging.warning("'nvidia-smi' not found.")
        return pandas.DataFrame(columns = cls.COLUMNS.keys()) # type: ignore[arg-type]

    @classmethod
    def count(cls, *, cache: bool = True) -> int:
        """
        Get the number of available GPUs.
        """
        return len(cls.get(cache = cache))

def main() -> None:

    parser = argparse.ArgumentParser(
        description = 'Print the list of detected GPUs.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--sep', type = str, required = False, default = ';', help = 'Separator between GPUs.')

    parser.add_argument('--cc', action = 'store_true', help = 'Return the compute capability as an integer.')

    args = parser.parse_args()

    if args.cc:
        gpus = (x.replace('.', '') for x in map(str, GPUDetector.get(enrich = False)['compute_cap']))
    else:
        gpus = (str(x) for x in GPUDetector.get(enrich = True)['architecture'])

    print(args.sep.join(gpus), end = '')

if __name__ == "__main__":

    main()
