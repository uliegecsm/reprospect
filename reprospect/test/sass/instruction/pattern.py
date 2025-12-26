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

    @staticmethod
    def opcode_mods(opcode: str, modifiers: typing.Iterable[int | str | None] | None = None) -> str:
        """
        Append each modifier with a `.`, within a proper named capture group.

        Note that the modifiers starting with a `?` are matched optionally.
        """
        opcode = PatternBuilder.group(opcode, group='opcode')
        if modifiers is None:
            return opcode

        parts: list[str] = [opcode]

        for modifier in modifiers:
            if not modifier:
                continue
            if isinstance(modifier, str) and modifier[0] == '?':
                parts.append(PatternBuilder.zero_or_one(r'\.' + PatternBuilder.group(modifier[1:], group='modifiers')))
            else:
                parts.append(r'\.' + PatternBuilder.group(modifier, group='modifiers'))

        return ''.join(parts)
