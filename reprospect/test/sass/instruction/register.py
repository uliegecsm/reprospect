import dataclasses
import typing

import regex

from reprospect.tools.sass.decode import RegisterType

REGISTER_MATCH : typing.Final[regex.Pattern[str]] = regex.compile(
    r'^(?P<rtype>(R|UR|P|UP))'
    r'((?P<special>(T|Z))|(?P<index>\d+))?'
    r'(?P<reuse>\.reuse)?$'
)

@dataclasses.dataclass(frozen = True, slots = True)
class RegisterMatch:
    rtype : RegisterType
    index : int
    reuse : bool = False

    @classmethod
    def parse(cls, value : str) -> 'RegisterMatch':
        """
        Parse an operand, assuming it is a register.

        >>> from reprospect.test.sass.instruction import RegisterMatch
        >>> RegisterMatch.parse('UR12')
        RegisterMatch(rtype=<RegisterType.UGPR: 'UR'>, index=12, reuse=False)
        """
        if (matched := REGISTER_MATCH.match(value)) is not None:
            if matched['index'] or matched['special']:
                return cls(
                    rtype = RegisterType(matched['rtype']),
                    index = int(matched['index']) if matched['index'] else -1,
                    reuse = bool(matched.group('reuse')),
                )
        raise ValueError(f'Invalid register format {value!r}.')
