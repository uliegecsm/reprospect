"""
Lightweight wrapper around CUDA driver and runtime libraries.

Provides a minimal interface for interacting with CUDA-capable devices and querying their properties.

.. note::

    This module has some functional overlap with ``cuda-bindings``. The API and the implementation may
    evolve to better complement or integrate with the NVIDIA CUDA Python bindings.
"""

import ctypes
import dataclasses
import functools
import logging
import typing

import cuda.bindings.driver # type: ignore[import-not-found]

from reprospect.tools import architecture

@dataclasses.dataclass(frozen = True, eq = True)
class CudaRuntimeError:
    """
    CUDA runtime error code.
    """
    value : int

    @property
    def success(self) -> bool:
        return self.value == 0

    def get(self, libcudart : ctypes.CDLL) -> tuple[str, str]:
        """
        Get error name and string.
        """
        libcudart.cudaGetErrorString.restype = ctypes.c_char_p
        libcudart.cudaGetErrorName.restype   = ctypes.c_char_p
        return libcudart.cudaGetErrorName(self.value).decode(), libcudart.cudaGetErrorString(self.value).decode()

@dataclasses.dataclass(frozen = True, eq = True)
class CudaDriverError:
    """
    CUDA driver error code.
    """
    value : int

    @property
    def success(self) -> bool:
        return self.value == 0

    def get(self, libcuda : ctypes.CDLL) -> tuple[str, str]:
        """
        Get error name and string.
        """
        error_msg  = ctypes.c_char_p()
        error_name = ctypes.c_char_p()
        libcuda.cuGetErrorString(self.value, ctypes.byref(error_msg))
        libcuda.cuGetErrorName  (self.value, ctypes.byref(error_name))

        assert error_name.value is not None
        assert error_msg.value is not None

        return error_name.value.decode(), error_msg.value.decode() # pylint: disable=no-member

class Cuda:

    libcuda   : ctypes.CDLL | None = None
    libcudart : ctypes.CDLL | None = None

    @classmethod
    def load(cls) -> None:
        """
        Load CUDA library.
        """
        if not cls.libcuda:
            cls.libcuda = ctypes.CDLL('libcuda.so')
            logging.info(f"Library {cls.libcuda} loaded successfully.")

        if not cls.libcudart:
            cls.libcudart = ctypes.CDLL('libcudart.so')
            logging.info(f"Library {cls.libcudart} loaded successfully.")

    @classmethod
    def check_driver_status(cls, *, status : CudaDriverError, info : typing.Any) -> None:
        """
        Check that `status` is successful, raise otherwise.
        """
        if not status.success:
            assert cls.libcuda is not None
            error_name, error_msg = status.get(libcuda = cls.libcuda)
            raise RuntimeError(
                f"{info} failed with error code {status} ({error_name}): {error_msg}",
            )

    @classmethod
    def check_runtime_status(cls, *, status : CudaRuntimeError, info : typing.Any) -> None:
        """
        Check that `status` is successful, raise otherwise.
        """
        if not status.success:
            assert cls.libcudart is not None
            error_name, error_msg = status.get(libcudart = cls.libcudart)
            raise RuntimeError(
                f"{info} failed with error code {status} ({error_name}): {error_msg}",
            )

    @classmethod
    def check_driver_api_call(cls, *, func : str) -> typing.Any:
        """
        Wrap CUDA driver API call `func` to raise an exception if the call is not successful.
        """
        assert cls.libcuda is not None
        handle = getattr(cls.libcuda, func)
        @functools.wraps(handle)
        def wrapper(*args, **kwargs):
            status = handle(*args, **kwargs)
            cls.check_driver_status(status = CudaDriverError(value = status), info = handle.__name__)
            return status
        return wrapper

    @classmethod
    def check_runtime_api_call(cls, *, func : str) -> typing.Any:
        """
        Wrap CUDA runtime API call `func` to raise an exception if the call is not successful.
        """
        assert cls.libcudart is not None
        handle = getattr(cls.libcudart, func)
        @functools.wraps(handle)
        def wrapper(*args, **kwargs):
            status = handle(*args, **kwargs)
            cls.check_runtime_status(status = CudaRuntimeError(value = status), info = handle.__name__)
            return status
        return wrapper

    def __init__(self, flags : int = 0) -> None:
        self.load()
        self.check_driver_api_call(func = 'cuInit')(flags)

    @functools.cached_property
    def device_count(self) -> int:
        """
        Wrap ``cudaGetDeviceCount``.
        """
        count = ctypes.c_int()
        self.check_runtime_api_call(func = 'cudaGetDeviceCount')(ctypes.byref(count))
        return count.value

    def get_device_attribute(self, *, value_type : type, attribute : cuda.bindings.driver.CUdevice_attribute, device : int) -> typing.Any:
        """
        Retrieve an attribute of `device`.
        """
        value = value_type()
        self.check_runtime_api_call(func = 'cudaDeviceGetAttribute')(ctypes.byref(value), attribute.value, device)
        return value.value

    def get_device_compute_capability(self, *, device : int) -> architecture.ComputeCapability:
        """
        Get compute capability of `device`.
        """
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        self.check_driver_api_call(func = 'cuDeviceComputeCapability')(
            ctypes.byref(cc_major),
            ctypes.byref(cc_minor),
            ctypes.c_int(device),
        )
        return architecture.ComputeCapability(major = cc_major.value, minor = cc_minor.value)

    def get_device_name(self, *, device : int, length : int = 150) -> str:
        """
        Get name of `device`.
        """
        name = b' ' * length
        self.check_driver_api_call(func = 'cuDeviceGetName')(ctypes.c_char_p(name), len(name), device)
        return name.split(b"\0", 1)[0].decode()

    def get_device_total_memory(self, *, device : int) -> int:
        """
        Get device total memory.
        """
        total_mem = ctypes.c_size_t()
        self.check_driver_api_call(func = 'cuDeviceTotalMem')(
            ctypes.byref(total_mem),
            ctypes.c_int(device),
        )
        return total_mem.value
