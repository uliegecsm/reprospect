import inspect
import sys
import unittest.mock

import pandas

from reprospect.tools import architecture
from reprospect.utils import detect

class TestGPUDetector:
    """
    Tests for :py:class:`reprospect.utils.detect.GPUDetector`.
    """
    OUTPUT = """\
index, uuid, name, compute_cap
0, GPU-12345678-1234-1234-1234-123456789012, NVIDIA GeForce RTX 3090, 8.6
1, GPU-87654321-4321-4321-4321-210987654321, NVIDIA A100-SXM4-40GB, 8.0
""".encode()

    def test_detect_gpus(self) -> None:
        """
        Test that :py:meth:`reprospect.utils.detect.GPUDetector.get` returns a properly formatted :py:class:`pandas.DataFrame`.
        """
        detect.GPUDetector.clear_cache()

        with unittest.mock.patch('shutil.which', side_effect = ['nvidia-smi']) as mock_shutil:
            with unittest.mock.patch('subprocess.check_output', side_effect = [self.OUTPUT]) as mock_subprocess:
                result = detect.GPUDetector.get(cache = True)

                assert isinstance(result, pandas.DataFrame)

                assert sorted(['index', 'uuid', 'name', 'compute_cap', 'architecture']) == sorted(result.columns)

                assert len(result) == 2
                assert detect.GPUDetector.count() == 2

                pandas.testing.assert_series_equal(result.iloc[0], pandas.Series({
                    'index' : 0,
                    'uuid' : 'GPU-12345678-1234-1234-1234-123456789012',
                    'name' : 'NVIDIA GeForce RTX 3090',
                    'compute_cap' : '8.6',
                    'architecture' : architecture.NVIDIAArch.from_str('AMPERE86'),
                }), check_names = False)

                pandas.testing.assert_series_equal(result.iloc[1], pandas.Series({
                    'index' : 1,
                    'uuid' : 'GPU-87654321-4321-4321-4321-210987654321',
                    'name' : 'NVIDIA A100-SXM4-40GB',
                    'compute_cap' : '8.0',
                    'architecture' : architecture.NVIDIAArch.from_str('AMPERE80'),
                }), check_names = False)

                mock_shutil.assert_called_once()

                mock_subprocess.assert_called_once_with(['nvidia-smi', '--query-gpu=uuid,index,name,compute_cap', '--format=csv'])

    def test_detect_gpus_no_gpu(self) -> None:
        """
        Test that :py:meth:`reprospect.utils.detect.GPUDetector.get` returns an empty :py:class:`pandas.DataFrame` if
        `nvidia-smi` is not found.
        """
        with unittest.mock.patch('shutil.which', side_effect = [None]) as mock_shutil:
            assert len(detect.GPUDetector.get(cache = False)) == 0

            mock_shutil.assert_called_once()

    def test_detect_gpus_cache(self) -> None:
        """
        Test that calling :py:meth:`reprospect.utils.detect.GPUDetector.get` twice
        uses caching and only calls `nvidia-smi` once.
        """
        detect.GPUDetector.clear_cache()

        with unittest.mock.patch('shutil.which', side_effect = ['nvidia-smi']) as mock_shutil:
            with unittest.mock.patch('subprocess.check_output', side_effect = [self.OUTPUT]) as mock_subprocess:
                result_first  = detect.GPUDetector.get(cache = True)
                result_second = detect.GPUDetector.get(cache = True)

                pandas.testing.assert_frame_equal(result_first, result_second)

                mock_subprocess.assert_called_once()
                mock_shutil.assert_called_once()

    def test_detect_gpus_no_cache(self) -> None:
        """
        Test that calling :py:meth:`reprospect.utils.detect.GPUDetector.get` twice
        when caching is disabled calls `nvidia-smi` twice.
        """
        detect.GPUDetector.clear_cache()

        with unittest.mock.patch('shutil.which', side_effect = ['nvidia-smi'] * 2) as mock_shutil:
            with unittest.mock.patch('subprocess.check_output', side_effect = [self.OUTPUT] * 2) as mock_subprocess:
                result_first  = detect.GPUDetector.get(cache = False)
                result_second = detect.GPUDetector.get(cache = False)

                pandas.testing.assert_frame_equal(result_first, result_second)

                assert mock_subprocess.call_count == 2
                assert mock_shutil.call_count == 2

class TestGPUDetectorAsScript:
    """
    Tests for :py:mod:`reprospect.utils.detect` in script mode.
    """
    SCRIPT = inspect.getfile(detect)

    def test(self, capsys):
        """
        Check that the output is correctly formatted according to the arguments.
        """
        with unittest.mock.patch(target = 'shutil.which', side_effect = ['nvidia-smi'] * 4) as mock_shutil:
            with unittest.mock.patch(target = 'subprocess.check_output', side_effect = [TestGPUDetector.OUTPUT] * 4) as mock_subprocess:
                # Compute capability, ';' separator.
                detect.GPUDetector.clear_cache()
                sys.argv = [self.SCRIPT, '--sep=;', '--cc']
                detect.main()
                assert capsys.readouterr().out == "86;80"

                # Compute capability, '/' separator.
                detect.GPUDetector.clear_cache()
                sys.argv = [self.SCRIPT, '--sep=/', '--cc']
                detect.main()
                assert capsys.readouterr().out == "86/80"

                # Named architecture, ',' separator.
                detect.GPUDetector.clear_cache()
                sys.argv = [__file__, '--sep=,']
                detect.main()
                assert capsys.readouterr().out == "AMPERE86,AMPERE80"

                # Named architecture, '|' separator.
                detect.GPUDetector.clear_cache()
                sys.argv = [__file__, '--sep=|']
                detect.main()
                assert capsys.readouterr().out == "AMPERE86|AMPERE80"

                assert mock_shutil.call_count == 4
                assert mock_subprocess.call_count == 4
