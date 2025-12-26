import typing

import pytest

from reprospect.test.sass.instruction.address import (
    AddressMatcher,
    GenericOrGlobalAddressMatch,
    LocalAddressMatch,
    SharedAddressMatch,
    StrideModifier,
)
from reprospect.test.sass.instruction.instruction import OpcodeModsWithOperandsMatcher
from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.test.sass.instruction.register import Register
from reprospect.tools.architecture import NVIDIAArch

from tests.parameters import PARAMETERS, Parameters


class TestAddressMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.address.AddressMatcher`.
    """
    def test_reg_address(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('TURING75')

        ADDRESS: typing.Final[str] = '[R42]'

        assert AddressMatcher(arch=ARCH           ).match(ADDRESS) == GenericOrGlobalAddressMatch(reg='R42')
        assert AddressMatcher(arch=ARCH, reg='R42').match(ADDRESS) == GenericOrGlobalAddressMatch(reg='R42')
        assert AddressMatcher(arch=ARCH, reg='R27').match(ADDRESS) is None

        assert AddressMatcher(arch=ARCH, memory=MemorySpace.LOCAL           ).match(ADDRESS) == LocalAddressMatch(reg='R42')
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.LOCAL, reg='R42').match(ADDRESS) == LocalAddressMatch(reg='R42')
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.LOCAL, reg='R27').match(ADDRESS) is None

        ADDRESS_WITH_OFFSET: typing.Final[str] = '[R42+0x10]'

        assert AddressMatcher(arch=ARCH                          ).match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10')
        assert AddressMatcher(arch=ARCH, reg='R42'               ).match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10')
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x10').match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10')
        assert AddressMatcher(arch=ARCH, reg='R27'               ).match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x20').match(ADDRESS_WITH_OFFSET) is None

        ADDRESS_WITH_NEG_OFFSET: typing.Final[str] = '[R208+-0x3fc0]'

        assert AddressMatcher(arch=ARCH                              ).match(ADDRESS_WITH_NEG_OFFSET) == GenericOrGlobalAddressMatch(reg='R208', offset='-0x3fc0')
        assert AddressMatcher(arch=ARCH, reg='R208'                  ).match(ADDRESS_WITH_NEG_OFFSET) == GenericOrGlobalAddressMatch(reg='R208', offset='-0x3fc0')
        assert AddressMatcher(arch=ARCH, reg='R208', offset='-0x3fc0').match(ADDRESS_WITH_NEG_OFFSET) == GenericOrGlobalAddressMatch(reg='R208', offset='-0x3fc0')
        assert AddressMatcher(arch=ARCH, reg='R27'                   ).match(ADDRESS_WITH_NEG_OFFSET) is None
        assert AddressMatcher(arch=ARCH, reg='R208', offset='0x3fc0' ).match(ADDRESS_WITH_NEG_OFFSET) is None

        ADDRESS_WITH_OFFSET_AS_UREG: typing.Final[str] = '[R42+UR4]'

        assert AddressMatcher(arch=ARCH                          ).match(ADDRESS_WITH_OFFSET_AS_UREG) == GenericOrGlobalAddressMatch(reg='R42', offset='UR4')
        assert AddressMatcher(arch=ARCH, reg='R42'               ).match(ADDRESS_WITH_OFFSET_AS_UREG) == GenericOrGlobalAddressMatch(reg='R42', offset='UR4')
        assert AddressMatcher(arch=ARCH, reg='R42', offset='UR4' ).match(ADDRESS_WITH_OFFSET_AS_UREG) == GenericOrGlobalAddressMatch(reg='R42', offset='UR4')
        assert AddressMatcher(arch=ARCH, reg='R27'               ).match(ADDRESS_WITH_OFFSET_AS_UREG) is None
        assert AddressMatcher(arch=ARCH, reg='R42', offset='UR2' ).match(ADDRESS_WITH_OFFSET_AS_UREG) is None
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x20').match(ADDRESS_WITH_OFFSET_AS_UREG) is None

    def test_reg64_address(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('AMPERE86')

        ADDRESS: typing.Final[str] = '[R42.64]'

        assert AddressMatcher(arch=ARCH           ).match(ADDRESS) == GenericOrGlobalAddressMatch(reg='R42')
        assert AddressMatcher(arch=ARCH, reg='R42').match(ADDRESS) == GenericOrGlobalAddressMatch(reg='R42')
        assert AddressMatcher(arch=ARCH, reg='R27').match(ADDRESS) is None

        ADDRESS_WITH_OFFSET: typing.Final[str] = '[R42.64+0x10]'

        assert AddressMatcher(arch=ARCH                          ).match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10')
        assert AddressMatcher(arch=ARCH, reg='R42'               ).match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10')
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x10').match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10')
        assert AddressMatcher(arch=ARCH, reg='R27'               ).match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x20').match(ADDRESS_WITH_OFFSET) is None

    def test_desc_reg64_address(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('BLACKWELL120')

        ADDRESS: typing.Final[str] = 'desc[UR4][R42.64]'

        assert AddressMatcher(arch=ARCH                            ).match(ADDRESS) == GenericOrGlobalAddressMatch(reg='R42', desc_ureg='UR4')
        assert AddressMatcher(arch=ARCH, reg='R42'                 ).match(ADDRESS) == GenericOrGlobalAddressMatch(reg='R42', desc_ureg='UR4')
        assert AddressMatcher(arch=ARCH, reg='R42', desc_ureg='UR4').match(ADDRESS) == GenericOrGlobalAddressMatch(reg='R42', desc_ureg='UR4')
        assert AddressMatcher(arch=ARCH, reg='R27'                 ).match(ADDRESS) is None
        assert AddressMatcher(arch=ARCH, reg='R42', desc_ureg='UR6').match(ADDRESS) is None

        ADDRESS_WITH_OFFSET: typing.Final[str] = 'desc[UR4][R42.64+0x10]'

        assert AddressMatcher(arch=ARCH                                           ).match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10', desc_ureg='UR4')
        assert AddressMatcher(arch=ARCH, reg='R42'                                ).match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10', desc_ureg='UR4')
        assert AddressMatcher(arch=ARCH, reg='R42',                desc_ureg='UR4').match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10', desc_ureg='UR4')
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x10', desc_ureg='UR4').match(ADDRESS_WITH_OFFSET) == GenericOrGlobalAddressMatch(reg='R42', offset='0x10', desc_ureg='UR4')
        assert AddressMatcher(arch=ARCH, reg='R27', offset='0x10'                 ).match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x10', desc_ureg='UR6').match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch=ARCH, reg='R42', offset='0x20', desc_ureg='UR4').match(ADDRESS_WITH_OFFSET) is None

    def test_reg_stride_address(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('VOLTA70')

        ADDRESS: typing.Final[str] = '[R25.X8+0x800]'

        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED                                                     ).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25'                                          ).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25', offset='0x800'                          ).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25', offset='0x800', stride=StrideModifier.X8).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25', offset='0x800', stride=StrideModifier.X4).match(ADDRESS) is None

    def test_shared_offset_address(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('VOLTA70')

        ADDRESS: typing.Final[str] = '[0x10]'

        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED           ).match(ADDRESS) == SharedAddressMatch(offset='0x10')
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25').match(ADDRESS) is None

        assert OpcodeModsWithOperandsMatcher(opcode='LDS', modifiers=('U', '128'), operands=(
            Register.REG, AddressMatcher.build_pattern(arch=ARCH, memory=MemorySpace.SHARED),
        )).match(inst=f'LDS.U.128 R20, {ADDRESS}') is not None

    def test_shared_rz_address(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('VOLTA70')

        assert OpcodeModsWithOperandsMatcher(opcode='LDS', modifiers=('U', '64'), operands=(
            Register.REG, AddressMatcher.build_pattern(arch=ARCH, memory=MemorySpace.SHARED),
        )).match(inst='LDS.U.64 R26, [RZ]') is not None

        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED               ).match('[RZ]') is not None
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, offset='0x10').match('[RZ]') is None
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25'    ).match('[RZ]') is None

    def test_shared_stride_16(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('TURING75')

        assert OpcodeModsWithOperandsMatcher(opcode='STS', modifiers=('128',), operands=(
            AddressMatcher.build_pattern(arch=ARCH, memory=MemorySpace.SHARED), Register.REG,
        )).match(inst='STS.128 [R49.X16], R16') is not None

    def test_shared_ur(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('HOPPER90')
        ADDRESS: typing.Final[str] = '[R32+UR10+-0x1c0]'

        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED           ).match(ADDRESS) == SharedAddressMatch(reg='R32', offset='UR10+-0x1c0')
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25').match(ADDRESS) is None

        assert OpcodeModsWithOperandsMatcher(opcode='LDS', modifiers=('128',), operands=(
            Register.REG, AddressMatcher.build_pattern(arch=ARCH, memory=MemorySpace.SHARED),
        )).match(inst=f'LDS.128 R32, {ADDRESS}') is not None

    def test_global_reg64_address_turing75(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('TURING75')

        assert OpcodeModsWithOperandsMatcher(opcode='LDG', modifiers=('E', '128', 'SYS'), operands=(
            Register.REG, AddressMatcher.build_pattern(arch=ARCH, memory=MemorySpace.GLOBAL),
        )).match(inst='LDG.E.128.SYS R24, [R24.64+UR4]') is not None

    def test_build_pattern(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('BLACKWELL120')

        assert AddressMatcher.build_pattern(arch=ARCH) == r'desc\[UR[0-9]+\]\[(?:R[0-9]+|UR[0-9]+)\.64(?:\+(?:-?0x[0-9A-Fa-f]+|UR[0-9]+|UR[0-9]+\+-?0x[0-9A-Fa-f]+))?\]'

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test(self, parameters: Parameters) -> None:
        """
        Ensure that :py:meth:`reprospect.test.sass.instruction.address.AddressMatcher.build_pattern` supports all
        :py:const:`tests.parameters.PARAMETERS`.
        """
        assert len(AddressMatcher.build_pattern(arch=parameters.arch)) > 0
