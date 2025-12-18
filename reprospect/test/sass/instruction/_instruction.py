from reprospect._impl import test  # type: ignore[import-not-found] # pylint: disable=no-name-in-module


def parse_any(instruction: str) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    str | None] | None:
    return test.sass._instruction.parse_any(instruction=instruction) # pylint: disable=protected-access
