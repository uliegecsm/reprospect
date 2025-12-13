![logo](docs/source/_static/logo.svg)

[![PyPI - Version](https://img.shields.io/pypi/v/reprospect?color=blue)](https://pypi.org/project/reprospect/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/reprospect)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/reprospect)

![PyPI - Downloads](https://img.shields.io/pypi/dd/reprospect)
![PyPI - Downloads](https://img.shields.io/pypi/dw/reprospect)
![PyPI - Downloads](https://img.shields.io/pypi/dm/reprospect)

# reprospect

`ReProspect` provides a framework and tools for reproducible prospecting of CUDA applications.

## Features

- 🧪 **Testing Framework**: Write tests for CUDA kernels with SASS (assembly) verification
- 🔍 **Binary Analysis**: Parse and analyze CUDA binaries (ELF, cubin) with cuobjdump and nvdisasm
- 📊 **Profiling Integration**: Interfaces for NVIDIA Nsight Compute and Nsight Systems
- 🎯 **SASS Decoding**: Decode and match GPU assembly instructions
- 🖥️ **Device Queries**: Detect and query CUDA-capable devices
- 🏗️ **CMake Integration**: Seamless integration with CMake-based projects

## Installation

```bash
pip install reprospect
```

### Requirements

- Python 3.10+
- CUDA Toolkit (for CUDA development features)
- CMake 3.27+ (for build system integration)

## Quick Start

### Detect GPUs

```python
from reprospect.utils.detect import GPUDetector

gpus = GPUDetector.get()
print(gpus)
```

### Analyze CUDA Binaries

```python
from reprospect.tools.binaries import CuObjDump
from reprospect.tools.architecture import NVIDIAArch

arch = NVIDIAArch.from_str('AMPERE86')
cuobjdump = CuObjDump.extract(
    file='my_program',
    arch=arch,
    sass=True
)
```

### Test SASS Instructions

```python
from reprospect.test.sass.instruction import instruction

# Match floating-point add instructions
matcher = instruction.Fp64AddMatcher()
matches = matcher.match(sass_instructions)
```

## Documentation

See [the full documentation](https://uliegecsm.github.io/reprospect/) for comprehensive guides, API reference, and examples.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the LGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
