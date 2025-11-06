import ctypes
import logging
import unittest

import cuda.bindings.driver # type: ignore[import-not-found]
import pytest

from reprospect.tools import device_properties
from reprospect.utils import detect

class TestCuda(unittest.TestCase):
    """
    Test for :py:class:`reprospect.tools.device_properties.Cuda`.
    """
    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_initialize(self) -> None:
        """
        Test it can be successfully loaded.
        """
        device_properties.Cuda.load()

        assert isinstance(device_properties.Cuda.libcuda,   ctypes.CDLL)
        assert isinstance(device_properties.Cuda.libcudart, ctypes.CDLL)

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_check_driver_status(self) -> None:
        """
        Check the proper handling of CUDA driver API error codes.
        """
        device_properties.Cuda.load()

        try:
            device_properties.Cuda.check_driver_status(status = device_properties.CudaDriverError(value = 100), info = 'My random information string.')
        except RuntimeError as e:
            assert str(e) == "My random information string. failed with error code CudaDriverError(value=100) (CUDA_ERROR_NO_DEVICE): no CUDA-capable device is detected"

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_check_runtime_status(self) -> None:
        """
        Check the proper handling of CUDA runtime API error codes.
        """
        device_properties.Cuda.load()

        try:
            device_properties.Cuda.check_runtime_status(status = device_properties.CudaRuntimeError(value = 201), info = 'My random information string.')
        except RuntimeError as e:
            assert str(e) == "My random information string. failed with error code CudaRuntimeError(value=201) (cudaErrorDeviceUninitialized): invalid device context"

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_device_count(self) -> None:
        """
        There must be at least one device.
        """
        assert device_properties.Cuda().device_count > 0

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_get_device_attribute(self) -> None:
        """
        Retrieve a few device attributes.
        """
        instance = device_properties.Cuda()

        for device in range(instance.device_count):
            logging.info(f'Looking at attributes of device {device}.')
            # For now, all Cuda devices have a 32 warp size.
            assert instance.get_device_attribute(value_type = ctypes.c_int, attribute = cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, device = device) == 32

            # For now, all Cuda devices have a 1024 max threads per block.
            assert instance.get_device_attribute(value_type = ctypes.c_int, attribute = cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device = device) == 1024

            # Get a few interesting attributes.
            attributes = [
                cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK,
                cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
            ]
            for attribute in attributes:
                logging.info(f'{attribute.name}: {instance.get_device_attribute(value_type = ctypes.c_int, attribute = cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device = device)}')

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_get_device_compute_capability(self) -> None:
        """
        Retrieve the compute capability of all devices available.
        """
        instance = device_properties.Cuda()

        for device in range(instance.device_count):
            cc = instance.get_device_compute_capability(device = device)
            logging.info(f'Compute capability of device {device} is {cc}.')

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_get_device_name(self) -> None:
        """
        Get the device name.
        """
        instance = device_properties.Cuda()

        for device in range(instance.device_count):
            cc = instance.get_device_name(device = device)
            assert isinstance(cc, str)
            logging.info(f'Name of device {device} is {cc}.')

    @pytest.mark.skipif(not detect.GPUDetector.count() > 0, reason = 'needs a GPU')
    def test_get_device_total_memory(self) -> None:
        """
        Get the device total memory.
        """
        instance = device_properties.Cuda()

        for device in range(instance.device_count):
            assert instance.get_device_total_memory(device = device) > 1024 * 1024 * 1024
