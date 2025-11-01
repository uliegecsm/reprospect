import ctypes
import dataclasses
import enum
import functools
import logging
import typing

import typeguard

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

    @typeguard.typechecked
    def get(self, cudart) -> tuple[str, str]:
        """
        Get error name and string.
        """
        cudart.cudaGetErrorString.restype = ctypes.c_char_p
        cudart.cudaGetErrorName.restype   = ctypes.c_char_p
        return cudart.cudaGetErrorName(self.value).decode(), cudart.cudaGetErrorString(self.value).decode()

@dataclasses.dataclass(frozen = True, eq = True)
class CudaDriverError:
    """
    CUDA driver error code.
    """
    value : int

    @property
    def success(self) -> bool:
        return self.value == 0

    @typeguard.typechecked
    def get(self, cuda) -> tuple[str, str]:
        """
        Get error name and string.
        """
        error_msg  = ctypes.c_char_p()
        error_name = ctypes.c_char_p()
        cuda.cuGetErrorString(self.value, ctypes.byref(error_msg))
        cuda.cuGetErrorName  (self.value, ctypes.byref(error_name))

        assert error_name.value is not None
        assert error_msg.value is not None

        return error_name.value.decode(), error_msg.value.decode() # pylint: disable=no-member

class CudaDeviceAttribute(enum.IntEnum):
    """
    CUDA device attribute, copied from ``cuda.h``.
    """
    MAX_THREADS_PER_BLOCK = 1
    MAX_BLOCK_DIM_X = 2
    MAX_BLOCK_DIM_Y = 3
    MAX_BLOCK_DIM_Z = 4
    MAX_GRID_DIM_X = 5
    MAX_GRID_DIM_Y = 6
    MAX_GRID_DIM_Z = 7
    MAX_SHARED_MEMORY_PER_BLOCK = 8
    SHARED_MEMORY_PER_BLOCK = 8
    TOTAL_CONSTANT_MEMORY = 9
    WARP_SIZE = 10
    MAX_PITCH = 11
    MAX_REGISTERS_PER_BLOCK = 12
    REGISTERS_PER_BLOCK = 12
    CLOCK_RATE = 13
    TEXTURE_ALIGNMENT = 14
    GPU_OVERLAP = 15
    MULTIPROCESSOR_COUNT = 16
    KERNEL_EXEC_TIMEOUT = 17
    INTEGRATED = 18
    CAN_MAP_HOST_MEMORY = 19
    COMPUTE_MODE = 20
    MAXIMUM_TEXTURE1D_WIDTH = 21
    MAXIMUM_TEXTURE2D_WIDTH = 22
    MAXIMUM_TEXTURE2D_HEIGHT = 23
    MAXIMUM_TEXTURE3D_WIDTH = 24
    MAXIMUM_TEXTURE3D_HEIGHT = 25
    MAXIMUM_TEXTURE3D_DEPTH = 26
    MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
    MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
    MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
    MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
    MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
    MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
    SURFACE_ALIGNMENT = 30
    CONCURRENT_KERNELS = 31
    ECC_ENABLED = 32
    PCI_BUS_ID = 33
    PCI_DEVICE_ID = 34
    TCC_DRIVER = 35
    MEMORY_CLOCK_RATE = 36
    GLOBAL_MEMORY_BUS_WIDTH = 37
    L2_CACHE_SIZE = 38
    MAX_THREADS_PER_MULTIPROCESSOR = 39
    ASYNC_ENGINE_COUNT = 40
    UNIFIED_ADDRESSING = 41
    MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
    MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
    CAN_TEX2D_GATHER = 44
    MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
    MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
    MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
    MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
    MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
    PCI_DOMAIN_ID = 50
    TEXTURE_PITCH_ALIGNMENT = 51
    MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
    MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
    MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
    MAXIMUM_SURFACE1D_WIDTH = 55
    MAXIMUM_SURFACE2D_WIDTH = 56
    MAXIMUM_SURFACE2D_HEIGHT = 57
    MAXIMUM_SURFACE3D_WIDTH = 58
    MAXIMUM_SURFACE3D_HEIGHT = 59
    MAXIMUM_SURFACE3D_DEPTH = 60
    MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
    MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
    MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
    MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
    MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
    MAXIMUM_SURFACECUBEMAP_WIDTH = 66
    MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
    MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
    MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
    MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
    MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
    MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
    MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
    MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
    COMPUTE_CAPABILITY_MAJOR = 75
    COMPUTE_CAPABILITY_MINOR = 76
    MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
    STREAM_PRIORITIES_SUPPORTED = 78
    GLOBAL_L1_CACHE_SUPPORTED = 79
    LOCAL_L1_CACHE_SUPPORTED = 80
    MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
    MAX_REGISTERS_PER_MULTIPROCESSOR = 82
    MANAGED_MEMORY = 83
    MULTI_GPU_BOARD = 84
    MULTI_GPU_BOARD_GROUP_ID = 85
    HOST_NATIVE_ATOMIC_SUPPORTED = 86
    SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87
    PAGEABLE_MEMORY_ACCESS = 88
    CONCURRENT_MANAGED_ACCESS = 89
    COMPUTE_PREEMPTION_SUPPORTED = 90
    CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91
    CAN_USE_STREAM_MEM_OPS_V1 = 92
    CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93
    CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94
    COOPERATIVE_LAUNCH = 95
    COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
    CAN_FLUSH_REMOTE_WRITES = 98
    HOST_REGISTER_SUPPORTED = 99
    PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
    DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101
    VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102
    VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
    HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
    HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104
    HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105
    MAX_BLOCKS_PER_MULTIPROCESSOR = 106
    GENERIC_COMPRESSION_SUPPORTED = 107
    MAX_PERSISTING_L2_CACHE_SIZE = 108
    MAX_ACCESS_POLICY_WINDOW_SIZE = 109
    GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110
    RESERVED_SHARED_MEMORY_PER_BLOCK = 111
    SPARSE_CUDA_ARRAY_SUPPORTED = 112
    READ_ONLY_HOST_REGISTER_SUPPORTED = 113
    TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114
    MEMORY_POOLS_SUPPORTED = 115
    GPU_DIRECT_RDMA_SUPPORTED = 116
    GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117
    GPU_DIRECT_RDMA_WRITES_ORDERING = 118
    MEMPOOL_SUPPORTED_HANDLE_TYPES = 119
    CLUSTER_LAUNCH = 120
    DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121
    CAN_USE_64_BIT_STREAM_MEM_OPS = 122
    CAN_USE_STREAM_WAIT_VALUE_NOR = 123
    DMA_BUF_SUPPORTED = 124
    IPC_EVENT_SUPPORTED = 125
    MEM_SYNC_DOMAIN_COUNT = 126
    TENSOR_MAP_ACCESS_SUPPORTED = 127
    HANDLE_TYPE_FABRIC_SUPPORTED = 128
    UNIFIED_FUNCTION_POINTERS = 129
    NUMA_CONFIG = 130
    NUMA_ID = 131
    MULTICAST_SUPPORTED = 132
    MPS_ENABLED = 133
    HOST_NUMA_ID = 134
    D3D12_CIG_SUPPORTED = 135
    MEM_DECOMPRESS_ALGORITHM_MASK = 136
    MEM_DECOMPRESS_MAXIMUM_LENGTH = 137
    GPU_PCI_DEVICE_ID    = 139
    GPU_PCI_SUBSYSTEM_ID = 140
    HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143

class Cuda:

    cuda   : ctypes.CDLL | None = None
    cudart : ctypes.CDLL | None = None

    @classmethod
    @typeguard.typechecked
    def load(cls) -> None:
        """
        Load CUDA library.
        """
        if not cls.cuda:
            cls.cuda = ctypes.CDLL('libcuda.so')
            logging.info(f"Library {cls.cuda} loaded successfully.")

        if not cls.cudart:
            cls.cudart = ctypes.CDLL('libcudart.so')
            logging.info(f"Library {cls.cudart} loaded successfully.")

    @classmethod
    @typeguard.typechecked
    def check_driver_status(cls, *, status : CudaDriverError, info : typing.Any) -> None:
        """
        Check that `status` is successful, raise otherwise.
        """
        if not status.success:
            error_name, error_msg = status.get(cuda = cls.cuda)
            raise RuntimeError(
                f"{info} failed with error code {status} ({error_name}): {error_msg}"
            )

    @classmethod
    @typeguard.typechecked
    def check_runtime_status(cls, *, status : CudaRuntimeError, info : typing.Any) -> None:
        """
        Check that `status` is successful, raise otherwise.
        """
        if not status.success:
            error_name, error_msg = status.get(cudart = cls.cudart)
            raise RuntimeError(
                f"{info} failed with error code {status} ({error_name}): {error_msg}"
            )

    @classmethod
    @typeguard.typechecked
    def check_driver_api_call(cls, *, func : str) -> typing.Any:
        """
        Wrap CUDA driver API call `func` to raise an exception if the call is not successful.
        """
        assert cls.cuda is not None
        handle = getattr(cls.cuda, func)
        @functools.wraps(handle)
        def wrapper(*args, **kwargs):
            status = handle(*args, **kwargs)
            cls.check_driver_status(status = CudaDriverError(value = status), info = handle.__name__)
            return status
        return wrapper

    @classmethod
    @typeguard.typechecked
    def check_runtime_api_call(cls, *, func : str) -> typing.Any:
        """
        Wrap CUDA runtime API call `func` to raise an exception if the call is not successful.
        """
        assert cls.cudart is not None
        handle = getattr(cls.cudart, func)
        @functools.wraps(handle)
        def wrapper(*args, **kwargs):
            status = handle(*args, **kwargs)
            cls.check_runtime_status(status = CudaRuntimeError(value = status), info = handle.__name__)
            return status
        return wrapper

    @typeguard.typechecked
    def __init__(self, flags : int = 0) -> None:
        self.load()
        self.check_driver_api_call(func = 'cuInit')(flags)

    @functools.cached_property
    @typeguard.typechecked
    def device_count(self) -> int:
        """
        Wrap ``cudaGetDeviceCount``.
        """
        count = ctypes.c_int()
        self.check_runtime_api_call(func = 'cudaGetDeviceCount')(ctypes.byref(count))
        return count.value

    @typeguard.typechecked
    def get_device_attribute(self, *, value_type : typing.Type, attribute : CudaDeviceAttribute, device : int) -> typing.Any:
        """
        Retrieve an attribute of `device`.
        """
        value = value_type()
        self.check_runtime_api_call(func = 'cudaDeviceGetAttribute')(ctypes.byref(value), attribute.value, device)
        return value.value

    @typeguard.typechecked
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

    @typeguard.typechecked
    def get_device_name(self, *, device : int, length : int = 150) -> str:
        """
        Get name of `device`.
        """
        name = b' ' * length
        self.check_driver_api_call(func = 'cuDeviceGetName')(ctypes.c_char_p(name), len(name), device)
        return name.split(b"\0", 1)[0].decode()

    @typeguard.typechecked
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
