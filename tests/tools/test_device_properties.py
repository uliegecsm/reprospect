import ctypes
import logging
import os
import pathlib
import re
import unittest

from cuda_helpers.tools import device_properties

class TestCuda(unittest.TestCase):
    """
    Test for :py:class:`toolbox.profiling.cuda.device_properties.Cuda`.
    """
    INCLUDE_DIR = pathlib.Path('/usr/local/cuda/include')

    def test_include_dir(self):
        """
        Check include directory.
        """
        assert self.INCLUDE_DIR.is_dir()
        assert (self.INCLUDE_DIR / 'cuda.h').is_file()

    def test_initialize(self):
        """
        Test it can be successfully loaded.
        """
        device_properties.Cuda.load()

        assert type(device_properties.Cuda.cuda)   is ctypes.CDLL
        assert type(device_properties.Cuda.cudart) is ctypes.CDLL

    def test_check_driver_status(self):
        """
        Check the proper handling of `Cuda` driver API error codes.
        """
        device_properties.Cuda.load()

        try:
            device_properties.Cuda.check_driver_status(status = device_properties.CudaDriverError(value = 100), info = 'My random information string.')
        except RuntimeError as e:
            assert str(e) == "My random information string. failed with error code CudaDriverError(value=100) (CUDA_ERROR_NO_DEVICE): no CUDA-capable device is detected"

    def test_check_runtime_status(self):
        """
        Check the proper handling of `Cuda` runtime API error codes.
        """
        device_properties.Cuda.load()

        try:
            device_properties.Cuda.check_runtime_status(status = device_properties.CudaRuntimeError(value = 201), info = 'My random information string.')
        except RuntimeError as e:
            assert str(e) == "My random information string. failed with error code CudaRuntimeError(value=201) (cudaErrorDeviceUninitialized): invalid device context"

    def test_device_count(self):
        """
        There must be at least one device.
        """
        assert device_properties.Cuda().device_count > 0

    def test_get_device_attribute(self):
        """
        Retrieve a few device attributes.
        """
        cuda = device_properties.Cuda()

        for device in range(cuda.device_count):
            logging.info(f'Looking at attributes of device {device}.')
            # For now, all Cuda devices have a 32 warp size.
            assert cuda.get_device_attribute(value_type = ctypes.c_int, attribute = device_properties.CudaDeviceAttribute.WARP_SIZE, device = device) == 32

            # For now, all Cuda devices have a 1024 max threads per block.
            assert cuda.get_device_attribute(value_type = ctypes.c_int, attribute = device_properties.CudaDeviceAttribute.MAX_THREADS_PER_BLOCK, device = device) == 1024

            # Get a few interesting attributes.
            attributes = [
                device_properties.CudaDeviceAttribute.MAX_REGISTERS_PER_MULTIPROCESSOR,
                device_properties.CudaDeviceAttribute.REGISTERS_PER_BLOCK,
                device_properties.CudaDeviceAttribute.TOTAL_CONSTANT_MEMORY,
            ]
            for attribute in attributes:
                logging.info(f'{attribute.name}: {cuda.get_device_attribute(value_type = ctypes.c_int, attribute = device_properties.CudaDeviceAttribute.TOTAL_CONSTANT_MEMORY, device = device)}')

    def test_get_device_compute_capability(self):
        """
        Retrieve the compute capability of all devices available.
        """
        cuda = device_properties.Cuda()

        for device in range(cuda.device_count):
            cc = cuda.get_device_compute_capability(device = device)
            assert isinstance(cc, tuple)
            logging.info(f'Compute capability of device {device} is {cc}.')

    def test_get_device_name(self):
        """
        Get the device name.
        """
        cuda = device_properties.Cuda()

        for device in range(cuda.device_count):
            cc = cuda.get_device_name(device = device)
            assert isinstance(cc, str)
            logging.info(f'Name of device {device} is {cc}.')

    def test_get_device_total_memory(sell):
        """
        Get the device total memory.
        """
        cuda = device_properties.Cuda()

        for device in range(cuda.device_count):
            assert cuda.get_device_total_memory(device = device) > 1024 * 1024 * 1024

    def test_cuda_device_attribute_uptodate(self):
        """
        Check that :py:enum:`CudaDeviceAttribute` is up-to-date with the content of `cuda.h`.
        """
        cuda_header = self.INCLUDE_DIR / 'cuda.h'

        assert cuda_header.is_file()

        to_be_found = set(rf'CU_DEVICE_ATTRIBUTE_{attribute.name}[ ]+=[ ]+{attribute.value}' for attribute in device_properties.CudaDeviceAttribute)

        with cuda_header.open('r') as fin:
            for line in fin:
                for found in [attr for attr in to_be_found if re.search(attr, line)]:
                    to_be_found.remove(found)

        assert len(to_be_found) == 0, to_be_found
