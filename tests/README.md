# Test Configuration

This directory contains separate configuration files for testing to allow more relaxed rules:

- **`.mypy.ini`**: Disables strict type checking for union attributes and indexing in tests
- **`.pylintrc`**: Disables documentation requirements and naming conventions for test code
- **`.pytest.ini`**: Configures pytest with strict mode and verbose output

These separate configs allow tests to:
- Use fixtures without "redefined-outer-name" warnings
- Skip docstrings for test functions (names are self-documenting)
- Use flexible naming conventions (e.g., single-letter variables in test data)

The main package code in `reprospect/` follows stricter rules defined at the repository root.
