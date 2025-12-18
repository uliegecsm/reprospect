from __future__ import annotations

import dataclasses
import typing

import attrs
import regex

from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.tools.architecture import NVIDIAArch


@dataclasses.dataclass(frozen = True, slots = True)
class AddressMatch:
    """
    Result of matching an address operand.
    """
    reg: str #: Register.
    offset: str | None = None #: Offset.
    ureg: str | None = None #: Uniform register holding the cache policy descriptor in a descriptor-based address.

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> AddressMatch:
        captured = bits.capturesdict()

        if not (value := captured.get('reg')):
            raise ValueError(bits)
        reg = value[0]

        offset: str | None = None
        if value := captured.get('offset'):
            if len(value) == 1:
                offset = value[0]

        ureg: str | None = None
        if value := captured.get('ureg'):
            if len(value) == 1:
                ureg = value[0]

        return cls(
            reg = reg,
            offset = offset,
            ureg = ureg,
        )

TEMPLATE_ADDRESS: typing.Final[str] = r'\[{reg}{offset}\]'
TEMPLATE_REG64_ADDRESS: typing.Final[str] = r'\[{reg}\.64{offset}\]'
TEMPLATE_DESC_REG64_ADDRESS: typing.Final[str] = r'desc\[{ureg}\]\[{reg}\.64{offset}\]'

@attrs.define(frozen = True, slots = True, kw_only = True)
class AddressMatcher:
    """
    Matcher for an address.
    """
    arch: NVIDIAArch
    ureg: str | None = None
    reg: str | None = None
    offset: str | bool | None = None

    pattern: regex.Pattern[str] = attrs.field(init = False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, 'pattern', regex.compile(self.build_pattern(
            arch = self.arch, ureg = self.ureg,
            reg = self.reg, offset = self.offset,
            captured = True,
        )))

    @classmethod
    def build_pattern_reg(cls, *, reg: str | None = None, captured: bool = True) -> str:
        pattern_reg = reg or rf'(?:{PatternBuilder.REG}|{PatternBuilder.UREG})'
        return PatternBuilder.group(pattern_reg, 'reg') if captured else pattern_reg

    @classmethod
    def build_pattern_offset(cls, *, offset: str | bool | None = None, captured: bool = True) -> str:
        if offset is None:
            return PatternBuilder.zero_or_one(r'\+' + (PatternBuilder.group(PatternBuilder.HEX, group = 'offset') if captured else PatternBuilder.HEX))
        if offset is False:
            return ''
        if offset is True:
            return rf'\+{PatternBuilder.group(PatternBuilder.HEX, "offset") if captured else PatternBuilder.HEX}'
        return rf'\+{PatternBuilder.group(offset, "offset") if captured else offset}'

    @classmethod
    def build_pattern_ureg(cls, *, ureg: str | None = None, captured: bool = True) -> str:
        pattern_ureg = ureg or PatternBuilder.UREG
        return PatternBuilder.group(pattern_ureg, 'ureg') if captured else pattern_ureg

    @classmethod
    def build_pattern(cls, *,
        arch: NVIDIAArch,
        ureg: str | None = None,
        reg: str | None = None,
        offset: str | bool | None = None,
        captured: bool = False,
    ) -> str:
        match arch.compute_capability.as_int:
            case 70 | 75:
                return cls.build_address(reg = reg, offset = offset, captured = captured)
            case 80 | 86 | 89:
                return cls.build_reg64_address(reg = reg, offset = offset, captured = captured)
            case 90 | 100 | 103 | 120:
                return cls.build_desc_reg64_address(reg = reg, ureg = ureg, offset = offset, captured = captured)
            case _:
                raise ValueError(f'unsupported architecture {arch}')

    @classmethod
    def build_address(cls, *, reg: str | None = None, offset: str | bool | None = None, captured: bool = False) -> str:
        """
        Basic address operand, such as::

            [R4]
            [R2+0x10]
        """
        reg = cls.build_pattern_reg(reg = reg, captured = captured)
        offset = cls.build_pattern_offset(offset = offset, captured = captured)
        return TEMPLATE_ADDRESS.format(reg = reg, offset = offset)

    @classmethod
    def build_reg64_address(cls, *, reg: str | None = None, offset: str | bool | None = None, captured: bool = False) -> str:
        """
        Address operand with ``.64`` modifier appended to the register, such as::

            [R1.64]
            [R2.64+0x10]
        """
        reg = cls.build_pattern_reg(reg = reg, captured = captured)
        offset = cls.build_pattern_offset(offset = offset, captured = captured)
        return TEMPLATE_REG64_ADDRESS.format(reg = reg, offset = offset)

    @classmethod
    def build_desc_reg64_address(cls, *, ureg: str | None = None, reg: str | None = None, offset: str | bool | None = None, captured: bool = False) -> str:
        """
        Address operand with cache policy descriptor, such as::

            desc[UR0][R0.64+0x10]
        """
        reg = cls.build_pattern_reg(reg = reg, captured = captured)
        offset = cls.build_pattern_offset(offset = offset, captured = captured)
        ureg = cls.build_pattern_ureg(ureg = ureg, captured = captured)
        return TEMPLATE_DESC_REG64_ADDRESS.format(reg = reg, offset = offset, ureg = ureg)

    def match(self, address: str) -> AddressMatch | None:
        if (matched := self.pattern.match(address)) is not None:
            return AddressMatch.parse(bits = matched)
        return None

    def __call__(self, address: str) -> AddressMatch | None:
        return self.match(address = address)
