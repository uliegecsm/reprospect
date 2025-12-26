import typing


class PatternBuilder:
    """
    Helper class to build patterns for instruction components.
    """
    HEX: typing.Final[str] = r'0x[0-9A-Fa-f]+'

    @classmethod
    def hex(cls) -> str:
        """
        :py:attr:`HEX` with `operands` group.
        """
        return cls.group(cls.HEX, group='operands')

    @staticmethod
    def zero_or_one(s: str) -> str:
        """
        Build an optional non-capturing pattern that matches zero or one occurrence of the given pattern.
        """
        return rf'(?:{s})?'

    @staticmethod
    def zero_or_more(s: str) -> str:
        """
        Build an optional non-capturing pattern that matches zero or more occurrences of the given pattern.
        """
        return rf'(?:{s})*'

    @staticmethod
    def any(*args: str) -> str:
        """
        Build a pattern matching any of `args`.
        """
        return f'(?:{"|".join(args)})'

    @staticmethod
    def group(s: int | str, group: str) -> str:
        """
        Wrap a pattern in a named capture group.
        """
        return rf'(?P<{group}>{s})'

    @staticmethod
    def groups(s: int | str, groups: typing.Iterable[str]) -> str:
        """
        Wrap a pattern in named capture groups.
        """
        for g in groups:
            s = PatternBuilder.group(s, group=g)
        return str(s)
