import functools
import typing


class PatternBuilder:
    """
    Helper class to build patterns for instruction components.
    """
    HEX: typing.Final[str] = r'0x[0-9A-Fa-f]+'
    OFFSET: typing.Final[str] = r'\+0x[0-9A-Fa-f]+'
    PRED: typing.Final[str] = r'P[0-9]+'
    REG: typing.Final[str] = r'R[0-9]+'
    REG64: typing.Final[str] = r'R[0-9]+\.64'
    UREG: typing.Final[str] = r'UR[0-9]+'

    #: Match a register or ``RZ``.
    REGZ: typing.Final[str] = r'R(?:Z|\d+)'

    #: Match a predicate register or ``PT``.
    PREDT: typing.Final[str] = r'P(?:T|\d+)'

    #: Match a uniform predicate register.
    UPRED: typing.Final[str] = r'UP[0-9]+'

    #: Match a uniform predicate register or ``UPT``.
    UPREDT: typing.Final[str] = r'UP(?:T|\d+)'

    OPERAND: typing.Final[str] = r'[\w@!\.\[\]\+\-\s]+'

    CONSTANT_BANK: typing.Final[str] = r'0x[0-9]+'
    """
    Constant memory bank.
    """

    CONSTANT_OFFSET: typing.Final[str] = r'(?:0x[0-9a-f]+|' + REG + '|' + UREG + ')'
    """
    Constant memory offset.
    """

    CONSTANT: typing.Final[str] = r'c\[' + CONSTANT_BANK + r'\]\[' + CONSTANT_OFFSET + r'\]'
    """
    Constant memory location.
    The bank looks like ``0x3`` while the address is either compile-time (*e.g.*
    ``0x899``) or depends on a register.
    """

    IMMEDIATE: typing.Final[str] = r'(-?\d+)(\.\d*)?((e|E)[-+]?\d+)?'
    """
    References:

    * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L34
    """

    PREDICATE: typing.Final[str] = r'@!?U?P(?:T|\d+)'
    """
    Predicate for the whole instruction (comes before the opcode).
    """

    PRE_OPERAND_MOD: typing.Final[str] = r'[\-!\|~]'
    """
    Allowed pre-modifiers for operands.

    References:

    * https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L67
    """

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
        return f'({"|".join(args)})'

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

    @classmethod
    def hex(cls) -> str:
        """
        :py:attr:`HEX` with `operands` group.
        """
        return cls.group(PatternBuilder.HEX, group='operands')

    @classmethod
    def reg(cls) -> str:
        """
        :py:attr:`REG` with `operands` group.
        """
        return cls.group(cls.REG, group='operands')

    @classmethod
    def premodregz(cls) -> str:
        """
        :py:attr:`REGZ` with `operands` group and optional :py:attr:`PRE_OPERAND_MOD` modifier.
        """
        return cls.group(cls.zero_or_one(cls.PRE_OPERAND_MOD) + cls.REGZ, group='operands')

    @classmethod
    def regz(cls) -> str:
        """
        :py:attr:`REGZ` with `operands` group.
        """
        return cls.group(cls.REGZ, group='operands')

    @classmethod
    def ureg(cls) -> str:
        """
        :py:attr:`UREG` with `operands` group.
        """
        return cls.group(cls.UREG, group='operands')

    @classmethod
    def anygpreg(cls, *, reuse: bool | None = None, group: str | None = None) -> str:
        """
        Match any general purpose register.
        """
        pattern = cls.any(cls.REG, cls.UREG)
        if reuse is None:
            pattern += PatternBuilder.zero_or_one(r'\.reuse')
        elif reuse is True:
            pattern += r'\.reuse'
        if group is not None:
            pattern = cls.group(pattern, group=group)
        return pattern

    @classmethod
    def predt(cls) -> str:
        """
        :py:attr:`PREDT` with `operands` group.
        """
        return cls.group(cls.PREDT, group='operands')

    @classmethod
    def immediate(cls) -> str:
        """
        :py:attr:`IMMEDIATE` with `operands` group.
        """
        return cls.group(cls.IMMEDIATE, group='operands')

    @classmethod
    def predicate(cls) -> str:
        """
        :py:attr:`PREDICATE` with `predicate` group.
        """
        return cls.group(s=cls.PREDICATE, group='predicate')

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

    @classmethod
    @functools.cache
    def operands(cls) -> str:
        """
        Many operands, with the `operands` named capture group.
        """
        op = PatternBuilder.group(cls.OPERAND, group='operands')
        return rf'{op}(?:\s*,\s*{op})*'
