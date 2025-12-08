import typing

from reprospect.tools.architecture import NVIDIAArch
from reprospect.test.sass.instruction.address import AddressMatch, AddressMatcher

class TestAddressMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.address.AddressMatcher`.
    """
    def test_address(self):
        ARCH : typing.Final[NVIDIAArch] = NVIDIAArch.from_str('VOLTA70')

        ADDRESS : typing.Final[str] = '[R42]'

        assert AddressMatcher(arch = ARCH             ).match(ADDRESS) == AddressMatch(reg = 'R42')
        assert AddressMatcher(arch = ARCH, reg = 'R42').match(ADDRESS) == AddressMatch(reg = 'R42')
        assert AddressMatcher(arch = ARCH, reg = 'R27').match(ADDRESS) is None

        ADDRESS_WITH_OFFSET : typing.Final[str] = '[R42+0x10]'

        assert AddressMatcher(arch = ARCH                              ).match(ADDRESS_WITH_OFFSET) == AddressMatch(reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, reg = 'R42'                 ).match(ADDRESS_WITH_OFFSET) == AddressMatch(reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, reg = 'R42', offset = '0x10').match(ADDRESS_WITH_OFFSET) == AddressMatch(reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, reg = 'R27'                 ).match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch = ARCH, reg = 'R42', offset = '0x20').match(ADDRESS_WITH_OFFSET) is None

    def test_reg64_address(self):
        ARCH : typing.Final[NVIDIAArch] = NVIDIAArch.from_str('AMPERE86')

        ADDRESS : typing.Final[str] = '[R42.64]'

        assert AddressMatcher(arch = ARCH             ).match(ADDRESS) == AddressMatch(reg = 'R42')
        assert AddressMatcher(arch = ARCH, reg = 'R42').match(ADDRESS) == AddressMatch(reg = 'R42')
        assert AddressMatcher(arch = ARCH, reg = 'R27').match(ADDRESS) is None

        ADDRESS_WITH_OFFSET : typing.Final[str] = '[R42.64+0x10]'

        assert AddressMatcher(arch = ARCH                              ).match(ADDRESS_WITH_OFFSET) == AddressMatch(reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, reg = 'R42'                 ).match(ADDRESS_WITH_OFFSET) == AddressMatch(reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, reg = 'R42', offset = '0x10').match(ADDRESS_WITH_OFFSET) == AddressMatch(reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, reg = 'R27'                 ).match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch = ARCH, reg = 'R42', offset = '0x20').match(ADDRESS_WITH_OFFSET) is None

    def test_desc_reg64_address(self):
        ARCH : typing.Final[NVIDIAArch] = NVIDIAArch.from_str('BLACKWELL120')

        ADDRESS : typing.Final[str] = 'desc[UR4][R42.64]'

        assert AddressMatcher(arch = ARCH                           ).match(ADDRESS) == AddressMatch(ureg = 'UR4', reg = 'R42')
        assert AddressMatcher(arch = ARCH,               reg = 'R42').match(ADDRESS) == AddressMatch(ureg = 'UR4', reg = 'R42')
        assert AddressMatcher(arch = ARCH, ureg = 'UR4', reg = 'R42').match(ADDRESS) == AddressMatch(ureg = 'UR4', reg = 'R42')
        assert AddressMatcher(arch = ARCH,               reg = 'R27').match(ADDRESS) is None
        assert AddressMatcher(arch = ARCH, ureg = 'UR6', reg = 'R42').match(ADDRESS) is None

        ADDRESS_WITH_OFFSET : typing.Final[str] = 'desc[UR4][R42.64+0x10]'

        assert AddressMatcher(arch = ARCH                                            ).match(ADDRESS_WITH_OFFSET) == AddressMatch(ureg = 'UR4', reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH,               reg = 'R42'                 ).match(ADDRESS_WITH_OFFSET) == AddressMatch(ureg = 'UR4', reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, ureg = 'UR4', reg = 'R42'                 ).match(ADDRESS_WITH_OFFSET) == AddressMatch(ureg = 'UR4', reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH, ureg = 'UR4', reg = 'R42', offset = '0x10').match(ADDRESS_WITH_OFFSET) == AddressMatch(ureg = 'UR4', reg = 'R42', offset = '0x10')
        assert AddressMatcher(arch = ARCH,               reg = 'R27', offset = '0x10').match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch = ARCH, ureg = 'UR6', reg = 'R42', offset = '0x10').match(ADDRESS_WITH_OFFSET) is None
        assert AddressMatcher(arch = ARCH, ureg = 'UR4', reg = 'R42', offset = '0x20').match(ADDRESS_WITH_OFFSET) is None

    def test_build_pattern(self):
        ARCH : typing.Final[NVIDIAArch] = NVIDIAArch.from_str('BLACKWELL120')

        pattern_address = AddressMatcher.build_pattern(arch = ARCH)
        assert pattern_address == r'desc\[UR[0-9]+\]\[(?:R[0-9]+|UR[0-9]+)\.64(?:\+0x[0-9A-Fa-f]+)?\]'
