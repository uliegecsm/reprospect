# Contributing to `ReProspect`

Thank you for your interest in `ReProspect`!
We welcome all contributions — whether you're fixing a bug, refining the documentation, or adding new features.

## Reporting bugs

If you find a bug, please [open an issue](https://github.com/uliegecsm/reprospect/issues) describing:

- What the bug is and how to reproduce it.
- What behavior you expected.

If you've already fixed the bug, feel free to [submit a pull request](https://github.com/uliegecsm/reprospect/pulls).

## Proposing features or improvements

Have an idea for a new feature, an enhancement, or a documentation update?
We'd love to hear about it!
Start a discussion by [opening an issue](https://github.com/uliegecsm/reprospect/issues) so we can plan the best approach together.

## Contributing examples

If you've created an example or demonstration that could help others, please share it by [opening a pull request](https://github.com/uliegecsm/reprospect/pulls).
Be sure to include a short description and usage notes.

## Development Guidelines

### Code Quality

- Follow PEP 8 style guidelines (enforced by `ruff` and `pylint`)
- Add type hints to all functions and methods
- Use `logging` instead of `print()` for library code (user-facing scripts and examples may use `print()`)
- Write docstrings for public APIs
- See [docs/CODE_QUALITY.md](docs/CODE_QUALITY.md) for notes on code patterns and linting exceptions

### Testing

- Write tests for new features
- Ensure existing tests pass before submitting
- Tests have separate, more relaxed linting rules (see `tests/README.md`)

### Documentation

- Update documentation for API changes
- Include examples for new features
- Documentation is built with Sphinx (see `docs/`)

---

Thank you for your interest in `ReProspect`!
