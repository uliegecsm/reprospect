"""
ReProspect: A framework for reproducible prospecting of CUDA applications.

ReProspect provides tools and utilities for:
- Testing and verifying CUDA kernel implementations
- Analyzing SASS (GPU assembly) code
- Profiling with NVIDIA Nsight Compute and Nsight Systems
- Querying GPU device properties
- Working with CUDA binaries (ELF, cubin, etc.)

Key modules:
- reprospect.test: Testing framework for CUDA code
- reprospect.tools: CUDA analysis and profiling tools
- reprospect.utils: Utility functions for CUDA development

For more information, see https://uliegecsm.github.io/reprospect/
"""

__version__ = "0.0.10"

__all__ = (
    '__version__',
)
