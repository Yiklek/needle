# Agent Guidelines - Needle Framework

This document provides guidelines for agentic coding agents working in the Needle deep learning framework repository.

## Build Commands

### Core Build
```bash
make lib          # Build C++ backend library (cmake + ninja)
make all          # Same as 'make lib'
make              # Default target (lib)
```

### Development Build
```bash
# Full clean build
make clean && make lib

# Python package installation with C++ backend
pip install -e .
```

### C++ Backend
```bash
# Manual CMake build
mkdir -p build
cd build && cmake .. -GNinja
cd build && ninja
cmake --install build --prefix python

# Install backend component only
cmake --install build --prefix python --component backend
```

## Lint & Format Commands

### Python
```bash
# Format Python code (Black)
make format                    # Formats all Python files
./scripts/format-py python     # Format specific directory
./scripts/format-py python --check --diff  # Check formatting only

# Lint Python code (Pylint)
./scripts/lint-py python       # Run Pylint
pylint python                  # Direct Pylint invocation
```

### C/C++
```bash
# Format C/C++ code (clang-format)
make format                    # Formats all C/C++ files  
./scripts/format-cc src        # Format C++ source files
./scripts/format-cc src "-n -Werror"  # Check formatting only

# Lint C/C++ code (clang-tidy)
./scripts/lint-cc src          # Run clang-tidy
TIDY_FLAGS="-quiet -warnings-as-errors=*" ./scripts/lint-cc src  # Strict mode
```

### Other
```bash
# CMake lint/format
cmake-lint CMakeLists.txt      # Lint CMake files
cmake-format CMakeLists.txt --check  # Check CMake formatting

# Shell script linting
shellcheck scripts/*           # Lint shell scripts
```

## Test Commands

### Python Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_nd_backend.py

# Run single test
pytest tests/test_nd_backend.py::test_function_name

# Run with coverage
pytest tests/ --cov=needle
```

### C++ Tests
```bash
# Build and run C++ tests
cd build && ctest              # Run CTest tests
./build/tests/test_tensor      # Run specific test executable
```

## Code Style Guidelines

### Python Style
- **Formatting**: Black with 120 character line length (see `pyproject.toml`)
- **Naming**:
  - `snake_case`: functions, variables, arguments, attributes, modules
  - `PascalCase`: classes  
  - `UPPER_CASE`: constants, class constants
  - `_single_leading_underscore`: "internal" identifiers
  - `single_trailing_underscore_`: avoid conflicts with Python keywords
- **Imports Order**: Standard library → third-party → local modules
- **Type Hints**: Use Python type hints where applicable
- **Docstrings**: Google style with Parameters/Returns sections
- **Error Handling**: Use specific exceptions, avoid bare `except:`

### C++ Style  
- **Formatting**: Google style with modifications (see `.clang-format`)
  - IndentWidth: 2 spaces
  - AccessModifierOffset: -2  
  - ColumnLimit: 120 characters
  - AllowShortFunctionsOnASingleLine: All
- **Naming** (see `.clang-tidy`):
  - `lower_case`: namespaces, variables, class members (suffix `_`)
  - `CamelCase`: classes, structs, template parameters, functions
  - `camelBack`: methods (exceptions: `New*`, `Get*`, `Register*`, `Clone*`)
  - `UPPER_CASE`: macros, constants
- **Headers**: Use `#pragma once`, include guards for portability
- **Error Handling**: Use exceptions judiciously, prefer RAII pattern

### File Organization
```
src/              # C++ source files
  ndarray_backend_cpu.cc
  ndarray_backend_cuda.cu
  fineflow/       # FineFlow submodule
  
python/needle/    # Python package
  __init__.py
  autograd.py     # Core autograd engine
  ops.py          # Operations
  nn.py           # Neural network modules
  backend_*/      # Backend implementations
  
tests/           # Test files
  test_*.py      # Python tests
  cpp/           # C++ tests
  
scripts/         # Utility scripts
  format-*       # Formatting scripts
  lint-*         # Linting scripts
```

## Import Conventions

### Python
```python
# Standard library
import os
import sys
from typing import List, Optional, Tuple, Union

# Third-party
import numpy as np
import torch

# Local modules
from needle import Tensor, ops
from needle.backend_selection import Device, NDArray
from . import submodule
```

### C++
```cpp
// System headers
#include <iostream>
#include <vector>
#include <memory>

// Third-party headers  
#include <pybind11/pybind11.h>

// Local headers
#include "fineflow/core/blob_tensor.h"
#include "fineflow/core/common/util.h"
```

## Error Handling Patterns

### Python
```python
# Use specific exceptions
try:
    result = operation()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except RuntimeError as e:
    # Handle runtime errors
    fallback_operation()
```

### C++
```cpp
# Prefer RAII for resource management
class Resource {
public:
    Resource() { acquire(); }
    ~Resource() { release(); }
    // No copy, allow move
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&&) = default;
    Resource& operator=(Resource&&) = default;
};

# Use exceptions for unrecoverable errors
if (!is_valid(input)) {
    throw std::invalid_argument("Invalid input");
}
```

## CI Integration

### GitHub Actions
- **Format Check**: Runs on push/PR to main/dev branches
- **Lint Check**: Includes Python, C++, CMake, shell script linting
- **Test Execution**: Use `pytest` for Python tests

### Local CI Simulation
```bash
# Run all checks locally
make format
./scripts/lint-py python
./scripts/lint-cc src
pytest tests/
```

## Development Workflow

1. **Make Changes**: Edit Python/C++ files
2. **Format Code**: `make format` or language-specific scripts
3. **Run Lints**: Use appropriate lint scripts
4. **Build**: `make lib` or `pip install -e .`
5. **Test**: Run relevant pytest tests
6. **Commit**: Follow existing commit message conventions

## Notes for Agents

- **Always run `make format`** before committing to ensure consistent formatting
- **Check both Python and C++ code** when making cross-language changes
- **Follow existing patterns** in similar files for consistency
- **Test changes** with `pytest` before considering work complete
- **Use type hints** in Python where they add clarity
- **Document public APIs** with Google-style docstrings