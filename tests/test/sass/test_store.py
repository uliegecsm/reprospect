import logging
import typing

import pytest

from reprospect.test import features
from reprospect.test.sass.instruction import StoreMatcher, StoreGlobalMatcher
from reprospect.tools.architecture import NVIDIAArch
from reprospect.utils import cmake

from tests.parameters                 import Parameters, PARAMETERS
from tests.test.sass.test_instruction import (
    get_decoder,
    CODE_ELEMENTWISE_ADD_RESTRICT,
    CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE,
    CODE_ELEMENTWISE_ADD_RESTRICT_256_WIDE,
)

class TestStoreMatcher:
    """
    Tests for :py:class:`reprospect.test.sass.instruction.StoreMatcher`
    and :py:class:`reprospect.test.sass.instruction.StoreGlobalMatcher`.
    """
    def test(self) -> None:
        matcher = StoreMatcher(arch = NVIDIAArch.from_compute_capability(86), size = 64, memory = None)
        assert matcher.match(inst = 'ST.E.64 [R4.64], R2') is not None

        matcher = StoreMatcher(arch = NVIDIAArch.from_compute_capability(100), size = 64, memory = None)
        assert matcher.match(inst = 'ST.E.64 desc[UR10][R4.64], R2') is not None

        matcher = StoreMatcher(arch = NVIDIAArch.from_compute_capability(120), size = 256, memory = 'G')
        assert matcher.match(inst = 'STG.E.ENL2.256 desc[UR4][R4.64], R8, R12') is not None

        matcher = StoreMatcher(arch = NVIDIAArch.from_compute_capability(86), size = 16, extend = 'U', memory = None)
        assert matcher.match(inst = 'ST.E.U16 [R2.64], R3') is not None

        matcher = StoreMatcher(arch = NVIDIAArch.from_compute_capability(86), size = 16, extend = 'S', memory = 'G')
        assert matcher.match(inst = 'STG.E.S16 [R2.64], R4') is not None

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_elementwise_add_restrict(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test store with :py:const:`tests.test.sass.test_instruction.CODE_ELEMENTWISE_ADD_RESTRICT`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the store.
        matcher = StoreGlobalMatcher(arch = parameters.arch)
        store = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(store) == 1

        inst, matched = store[0]

        logging.info(f'{matcher} matched instruction {inst.instruction} as {matched}.')

        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_elementwise_add_restrict_128_wide(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test 128-bit wide store with
        :py:const:`tests.test.sass.test_instruction.CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_128_WIDE)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        # Find the wide store.
        matcher = StoreGlobalMatcher(arch = parameters.arch, size = 128)
        store = [(inst, matched) for inst in decoder.instructions if (matched := matcher.match(inst))]
        assert len(store) == 1

        inst, matched = store[0]

        logging.info(f'{matcher} matched instruction {inst.instruction} as {matched}.')

        assert '128' in matched.modifiers
        assert len(matched.additional['address']) == 1
        assert len(matched.operands) == 2

    @pytest.mark.parametrize('parameters', PARAMETERS, ids = str)
    def test_elementwise_add_restrict_256_wide(self, request, workdir, parameters : Parameters, cmake_file_api : cmake.FileAPI) -> None:
        """
        Test 256-bit wide store with
        :py:const:`tests.test.sass.test_instruction.CODE_ELEMENTWISE_ADD_RESTRICT_256_WIDE`.
        """
        FILE = workdir / f'{request.node.originalname}.{parameters.arch.as_sm}.cu'
        FILE.write_text(CODE_ELEMENTWISE_ADD_RESTRICT_256_WIDE)

        decoder, _ = get_decoder(cwd = workdir, arch = parameters.arch, file = FILE, cmake_file_api = cmake_file_api)

        aligned_16 : typing.Final[bool] = features.Memory(arch = parameters.arch).max_transaction_size == 16

        # Find the wide stores(s).
        matcher_s_128 = StoreGlobalMatcher(arch = parameters.arch, size = 128)
        matcher_s_256 = StoreGlobalMatcher(arch = parameters.arch, size = 256)
        s_128 = tuple(filter(matcher_s_128, decoder.instructions))
        s_256 = tuple(filter(matcher_s_256, decoder.instructions))
        if aligned_16:
            assert len(s_128) == 2 and len(s_256) == 0
        else:
            assert len(s_128) == 0 and len(s_256) == 1
