import typing

import pytest

from reprospect.test.sass.instruction.address import (
    AddressMatcher,
    GenericOrGlobalAddressMatch,
    LocalAddressMatch,
    SharedAddressMatch,
    StrideModifier,
)
from reprospect.test.sass.instruction.memory import MemorySpace
from reprospect.tools.architecture import NVIDIAArch

from tests.parameters import PARAMETERS, Parameters


class TestAddressMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.address.AddressMatcher`.
    """
    def test_address(self) -> None:
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

    def test_stride_address(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('VOLTA70')

        ADDRESS: typing.Final[str] = '[R25.X8+0x800]'

        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED                                                     ).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25'                                          ).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25', offset='0x800'                          ).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25', offset='0x800', stride=StrideModifier.X8).match(ADDRESS) == SharedAddressMatch(reg='R25', offset='0x800', stride=StrideModifier.X8)
        assert AddressMatcher(arch=ARCH, memory=MemorySpace.SHARED, reg='R25', offset='0x800', stride=StrideModifier.X4).match(ADDRESS) is None

    def test_build_pattern(self) -> None:
        ARCH: typing.Final[NVIDIAArch] = NVIDIAArch.from_str('BLACKWELL120')

        assert AddressMatcher.build_pattern(arch=ARCH) == r'desc\[UR[0-9]+\]\[(?:R[0-9]+|UR[0-9]+)\.64(?:\+(?:-?0x[0-9A-Fa-f]+|UR[0-9]+))?\]'

    @pytest.mark.parametrize('parameters', PARAMETERS, ids=str)
    def test(self, parameters: Parameters) -> None:
        """
        Ensure that :py:meth:`reprospect.test.sass.instruction.address.AddressMatcher.build_pattern` supports all
        :py:const:`tests.parameters.PARAMETERS`.
        """
        assert len(AddressMatcher.build_pattern(arch=parameters.arch)) > 0
