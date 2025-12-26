from __future__ import annotations

import dataclasses
import sys
import typing

import attrs
import regex

from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.test.sass.instruction.pattern import PatternBuilder
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import Self
else:
    from backports.strenum.strenum import StrEnum
    from typing_extensions import Self

MODIFIER_STRIDE: typing.Final[str] = r'X(?:4|8)'
"""Stride modifier."""

class StrideModifier(StrEnum):
    """
    Stride modifier.
    """
    X4 = 'X4'
    X8 = 'X8'

@dataclasses.dataclass(frozen=True, slots=True)
class GenericOrGlobalAddressMatch:
    """
    Generic or global address operand, such as::

        desc[UR42][R8+0xf1]
    """
    reg: str
    offset: str | None = None
    desc_ureg: str | None = None
    """:py:attr:`reprospect.tools.sass.decode.RegisterType.UGPR` from cache policy descriptor."""

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> GenericOrGlobalAddressMatch:
        if not (reg := bits.group('reg')):
            raise ValueError(bits)

        captured_dict = bits.capturesdict()

        offset: str | None = None
        if (value := captured_dict.get('offset')) and len(value) == 1:
            offset = value[0]

        desc_ureg: str | None = None
        if (value := captured_dict.get('desc_ureg')) and len(value) == 1:
            desc_ureg = value[0]

        return cls(reg=reg, offset=offset, desc_ureg=desc_ureg)

@dataclasses.dataclass(frozen=True, slots=True)
class SharedAddressMatch:
    """
    Shared address operand, such as::

        [R25.X8+0x800]
        [0x10]
    """
    reg: str | None = None
    offset: str | None = None
    stride: StrideModifier | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> SharedAddressMatch:
        captured_dict = bits.capturesdict()

        reg: str | None = None
        if (value := captured_dict.get('reg')) and len(value) == 1:
            reg = value[0]

        offset: str | None = None
        if (value := captured_dict.get('offset')) and len(value) == 1:
            offset = value[0]

        stride: StrideModifier | None = None
        if (value := captured_dict.get('stride')) and len(value) == 1:
            stride = StrideModifier(value[0])

        return cls(reg=reg, offset=offset, stride=stride)

@dataclasses.dataclass(frozen=True, slots=True)
class LocalAddressMatch:
    """
    Local address operand, such as::

        [R47]
    """
    reg: str
    offset: str | None = None

    @classmethod
    def parse(cls, bits: regex.Match[str]) -> Self:
        if not (reg := bits.group('reg')):
            raise ValueError(bits)

        offset: str | None = None
        if (value := bits.capturesdict().get('offset')) and len(value) == 1:
            offset = value[0]

        return cls(reg=reg, offset=offset)

TEMPLATE_REG_ADDRESS: typing.Final[str] = r'\[{reg}{offset}\]'
TEMPLATE_REG_STRIDE_ADDRESS: typing.Final[str] = r'\[{reg}{stride}{offset}\]'
TEMPLATE_REG64_ADDRESS: typing.Final[str] = r'\[{reg}\.64{offset}\]'
TEMPLATE_DESC_REG64_ADDRESS: typing.Final[str] = r'desc\[{desc_ureg}\]\[{reg}\.64{offset}\]'

@attrs.define(frozen=True, slots=True, kw_only=True)
class AddressMatcher:
    """
    Matcher for an address.
    """
    arch: NVIDIAArch
    memory: MemorySpace = MemorySpace.GENERIC
    reg: str | None = None
    offset: str | None = None
    desc_ureg: str | None = None
    stride: StrideModifier | None = None

    pattern: regex.Pattern[str] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, 'pattern', regex.compile(self.build_pattern(
            arch=self.arch, memory=self.memory,
            reg=self.reg, offset=self.offset,
            desc_ureg=self.desc_ureg, stride=self.stride,
            captured=True,
        )))

    @classmethod
    def build_pattern_reg(cls, *, arch: NVIDIAArch, reg: str | None = None, captured: bool = True) -> str:
        pattern_reg = reg or (rf'(?:{Register.REG}|{Register.UREG})' if arch.compute_capability.as_int >= 75 else Register.REG)
        return PatternBuilder.group(pattern_reg, 'reg') if captured else pattern_reg

    @classmethod
    def build_pattern_offset(cls, *, arch: NVIDIAArch, offset: str | None = None, captured: bool = True) -> str:
        if offset is not None:
            return rf'\+{PatternBuilder.group(offset, "offset") if captured else offset}'
        inner = rf'(?:-?{PatternBuilder.HEX}|{Register.UREG})' if arch.compute_capability.as_int >= 75 else rf'-?{PatternBuilder.HEX}'
        return PatternBuilder.zero_or_one(r'\+' + (PatternBuilder.group(inner, 'offset') if captured else inner))

    @classmethod
    def build_pattern_desc_ureg(cls, *, desc_ureg: str | None = None, captured: bool = True) -> str:
        pattern_desc_ureg = desc_ureg or Register.UREG
        return PatternBuilder.group(pattern_desc_ureg, 'desc_ureg') if captured else pattern_desc_ureg

    @classmethod
    def build_pattern_stride(cls, *, stride: StrideModifier | None = None, captured: bool = True) -> str:
        if stride is not None:
            inner = PatternBuilder.group(stride.value, 'stride') if captured else stride.value
            return rf'\.{inner}'
        inner = PatternBuilder.group(MODIFIER_STRIDE, 'stride') if captured else MODIFIER_STRIDE
        return PatternBuilder.zero_or_one(rf'\.{inner}')

    @classmethod
    def build_pattern(cls, *,
        arch: NVIDIAArch,
        memory: MemorySpace = MemorySpace.GENERIC,
        reg: str | None = None,
        offset: str | None = None,
        desc_ureg: str | None = None,
        stride: StrideModifier | None = None,
        captured: bool = False,
    ) -> str:
        match memory:
            case MemorySpace.GENERIC | MemorySpace.GLOBAL:
                return cls.build_generic_or_global_address(arch=arch, reg=reg, offset=offset, desc_ureg=desc_ureg, captured=captured)
            case MemorySpace.SHARED:
                return cls.build_shared_address(arch=arch, reg=reg, stride=stride, offset=offset, captured=captured)
            case MemorySpace.LOCAL:
                return cls.build_local_address(arch=arch, reg=reg, offset=offset, captured=captured)
            case _:
                raise ValueError(f'unsupported memory space {memory}')

    @classmethod
    def build_generic_or_global_address(cls, *, arch: NVIDIAArch, reg: str | None = None, offset: str | None = None, desc_ureg: str | None = None, captured: bool = False) -> str:
        """
        Generic or global memory address operand.
        """
        match arch.compute_capability.as_int:
            case 70:
                return cls.build_reg_address(arch=arch, reg=reg, offset=offset, captured=captured)
            case 75:
                return PatternBuilder.any(
                    cls.build_reg_address(arch=arch, reg=reg, offset=offset, captured=captured),
                    cls.build_reg64_address(arch=arch, reg=reg, offset=offset, captured=captured),
                )
            case 80 | 86 | 89:
                return cls.build_reg64_address(arch=arch, reg=reg, offset=offset, captured=captured)
            case 90 | 100 | 103 | 120:
                return cls.build_desc_reg64_address(arch=arch, reg=reg, desc_ureg=desc_ureg, offset=offset, captured=captured)
            case _:
                raise ValueError(f'unsupported architecture {arch}')

    @classmethod
    def build_shared_address(cls, *, arch: NVIDIAArch, reg: str | None = None, offset: str | None = None, stride: StrideModifier | None = None, captured: bool = False) -> str:
        """
        Shared memory address operand.
        """
        reg_stride_address = cls.build_reg_stride_address(arch=arch, reg=reg, offset=offset, stride=stride, captured=captured)
        if reg is not None or stride is not None:
            return reg_stride_address
        offset_address = offset or PatternBuilder.HEX
        offset_address = rf"\[{PatternBuilder.group(offset_address, group='offset') if captured else offset_address}\]"

        return PatternBuilder.any(reg_stride_address, offset_address)

    @classmethod
    def build_local_address(cls, *, arch: NVIDIAArch, reg: str | None = None, offset: str | None = None, captured: bool = False) -> str:
        """
        Local memory address operand.
        """
        return cls.build_reg_address(arch=arch, reg=reg, offset=offset, captured=captured)

    @classmethod
    def build_reg_address(cls, *, arch: NVIDIAArch, reg: str | None = None, offset: str | None = None, captured: bool = False) -> str:
        """
        Basic address operand, such as::

            [R4]
            [R2+0x10]
        """
        pattern_reg = cls.build_pattern_reg(arch=arch, reg=reg, captured=captured)
        pattern_offset = cls.build_pattern_offset(arch=arch, offset=offset, captured=captured)
        return TEMPLATE_REG_ADDRESS.format(reg=pattern_reg, offset=pattern_offset)

    @classmethod
    def build_reg_stride_address(cls, *, arch: NVIDIAArch, reg: str | None = None, offset: str | None = None, stride: StrideModifier | None = None, captured: bool = False) -> str:
        """
        Address operand with stride modifier, such as::

            [R2.X8+0x10]
        """
        pattern_reg = cls.build_pattern_reg(arch=arch, reg=reg, captured=captured)
        pattern_offset = cls.build_pattern_offset(arch=arch, offset=offset, captured=captured)
        pattern_stride = cls.build_pattern_stride(stride=stride, captured=captured)
        return TEMPLATE_REG_STRIDE_ADDRESS.format(reg=pattern_reg, stride=pattern_stride, offset=pattern_offset)

    @classmethod
    def build_reg64_address(cls, *, arch: NVIDIAArch, reg: str | None = None, offset: str | None = None, captured: bool = False) -> str:
        """
        Address operand with ``.64`` modifier appended to the register, such as::

            [R1.64]
            [R2.64+0x10]
            [R14.64+UR6]
        """
        pattern_reg = cls.build_pattern_reg(arch=arch, reg=reg, captured=captured)
        pattern_offset = cls.build_pattern_offset(arch=arch, offset=offset, captured=captured)
        return TEMPLATE_REG64_ADDRESS.format(reg=pattern_reg, offset=pattern_offset)

    @classmethod
    def build_desc_reg64_address(cls, *, arch: NVIDIAArch, reg: str | None = None, offset: str | None = None, desc_ureg: str | None = None, captured: bool = False) -> str:
        """
        Address operand with cache policy descriptor, such as::

            desc[UR0][R0.64+0x10]
        """
        pattern_reg = cls.build_pattern_reg(arch=arch, reg=reg, captured=captured)
        pattern_offset = cls.build_pattern_offset(arch=arch, offset=offset, captured=captured)
        pattern_desc_ureg = cls.build_pattern_desc_ureg(desc_ureg=desc_ureg, captured=captured)
        return TEMPLATE_DESC_REG64_ADDRESS.format(reg=pattern_reg, offset=pattern_offset, desc_ureg=pattern_desc_ureg)

    def match(self, address: str) -> GenericOrGlobalAddressMatch | SharedAddressMatch | LocalAddressMatch | None:
        if (matched := self.pattern.match(address)) is not None:
            match self.memory:
                case MemorySpace.GENERIC | MemorySpace.GLOBAL:
                    return GenericOrGlobalAddressMatch.parse(bits=matched)
                case MemorySpace.SHARED:
                    return SharedAddressMatch.parse(bits=matched)
                case MemorySpace.LOCAL:
                    return LocalAddressMatch.parse(bits=matched)
                case _:
                    raise ValueError(f'unsupported memory space {self.memory}')
        return None

    def __call__(self, address: str) -> GenericOrGlobalAddressMatch | SharedAddressMatch | LocalAddressMatch | None:
        return self.match(address=address)
