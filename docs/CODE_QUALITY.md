# Code Quality Notes

This document explains certain code patterns and linting exceptions in the codebase.

## Pylint Disable Comments

The codebase contains 29 `pylint: disable` comments. Here's why they're needed:

### Legitimate Library Issues (3 occurrences)

**`blake3.blake3()` - not-callable**
- Files: `reprospect/tools/cacher.py`, `reprospect/tools/nsys.py`, `reprospect/tools/ncu/cacher.py`
- Reason: Known issue with blake3's type stubs; the function is callable despite pylint's warning
- Action: Keep disable comments

**`error_name.value.decode()` - no-member**
- File: `reprospect/tools/device_properties.py`
- Reason: ctypes dynamic attribute access; pylint can't infer c_char_p members
- Action: Keep disable comment

### Intentional Design Choices (8+ occurrences)

**`invalid-name` for Constants**
- Used for: `CMD`, `ALLOWABLE_SIZES`, type alias shortcuts (e.g., `Instructions`, `Requests`)
- Reason: Following convention that constants are uppercase, even if short
- Action: Keep disable comments - this is a valid style choice

**`too-many-instance-attributes` (2 occurrences)**
- Files: `reprospect/tools/nsys.py:35`, `reprospect/tools/ncu/session.py:14`
- Classes: `Command` configuration classes
- Reason: Command-line tools naturally have many configuration options (10+ attributes)
- Action: Keep - these classes represent complex configurations and shouldn't be split

**`duplicate-code` (1 occurrence)**
- File: `reprospect/tools/nsys.py:35`
- Reason: Similar Command classes in nsys and ncu have similar structure by design
- Action: Keep - parallelism in design is intentional

**`too-many-branches` / `too-many-return-statements`**
- Files: Pattern matching and instruction parsing code
- Reason: Complex state machines and parsers naturally have many branches
- Action: Keep - splitting would make code less readable

**`unused-argument` in abstract methods**
- File: `reprospect/test/sass/composite_impl.py:48`
- Reason: Interface definition requires the parameter for subclasses
- Action: Keep - this is standard for abstract interfaces

## Type Annotations

The codebase has excellent type annotation coverage. The 3 `type: ignore` comments are all legitimate:

1. **`cuda.bindings.driver` import** - External CUDA library without complete stubs
2. **Dynamic ctypes attributes** - Can't be statically typed
3. **Complex mypy inference issues** - Where the type is correct but mypy can't infer it

## Print Statements

The codebase properly uses `logging` throughout library code. The few `print()` statements are in appropriate places:

- CLI tools (`reprospect/utils/detect.py:93`) - Needs stdout output
- Test helpers - Creating test output
- Rich library printing - Using rich.Console for formatted output

## Recommendations

1. ✅ Current pylint disables are justified
2. ✅ Type annotations are comprehensive
3. ✅ Logging is used appropriately
4. 📝 Consider adding this document link in CONTRIBUTING.md for new contributors
