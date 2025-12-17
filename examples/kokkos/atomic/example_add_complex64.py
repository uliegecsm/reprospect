import logging
import re
import sys
import typing

from reprospect.test.sass.composite import instruction_is, unordered_interleaved_instructions_are, interleaved_instructions_are
from reprospect.test.sass.composite_impl import SequenceMatcher, UnorderedInterleavedInSequenceMatcher
from reprospect.test.sass.controlflow.block import BasicBlockMatcher, BasicBlockWithParentMatcher
from reprospect.test.sass.instruction import (
    AtomicMatcher,
    Fp32AddMatcher,
    InstructionMatch,
    LoadGlobalMatcher,
    RegisterMatcher,
)
from reprospect.tools.architecture import NVIDIAArch
from reprospect.tools.sass import Decoder, ControlFlow
from reprospect.tools.sass.controlflow import Graph

from examples.kokkos.atomic import add

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class Operation(typing.Protocol):
    def build(self, load: InstructionMatch) -> SequenceMatcher:
        ...

class AtomicCAS:
    def __init__(self, arch: NVIDIAArch, operation: Operation) -> None:
        self.arch: typing.Final[NVIDIAArch] = arch
        self.operation: typing.Final[Operation] = operation

    def match(self, cfg: Graph) -> None:
        # First, find the global load from memory.
        matcher_ldg = LoadGlobalMatcher(arch=self.arch, size=64, readonly=False)
        block_ldg, [matched_ldg] = BasicBlockMatcher(matcher=matcher_ldg).assert_matches(cfg=cfg)
        logging.info(matched_ldg)

        # Then, match the operation with the CAS.
        matcher_op = self.operation.build(load=matched_ldg)
        matcher_cas = instruction_is(AtomicMatcher(
            arch=self.arch,
            operation='CAS',
            dtype=(None, 64),
            scope='DEVICE',
            consistency='STRONG',
            memory='',
        )).with_operand(index=3, operand=matched_ldg.operands[0])

        _, matched = BasicBlockWithParentMatcher(parent=block_ldg, matcher=interleaved_instructions_are(matcher_op, matcher_cas)).assert_matches(cfg=cfg)

        assert len(matched) == 3

        assert matched[2].operands[4] in (matched[0].operands[0], matched[1].operands[1])

class AddComplex64:
    """
    Addition of two 64-bit complex values.
    """
    def build(self, load: InstructionMatch) -> UnorderedInterleavedInSequenceMatcher:

        assert (load_reg := RegisterMatcher(special=False).match(reg=load.operands[0])) is not None
        assert load_reg.index is not None

        reg_real = f'{load_reg.rtype}{load_reg.index}'
        reg_imag = f'{load_reg.rtype}{load_reg.index + 1}'

        return unordered_interleaved_instructions_are(
            instruction_is(Fp32AddMatcher()).with_operand(index=1, operand=reg_real),
            instruction_is(Fp32AddMatcher()).with_operand(index=1, operand=reg_imag),
        )

class TestAtomicAddComplex64(add.TestCase):
    """
    Verify that :code:`Kokkos::atomic_add` for :code:`Kokkos::complex<float>` maps to an atomic CAS-based implementation.
    """
    @classmethod
    @override
    def get_target_name(cls) -> str:
        return 'examples_kokkos_atomic_add_complex64'

    SIGNATURE_MATCHER: typing.ClassVar[re.Pattern[str]] = re.compile(
        r'AtomicAddFunctor<Kokkos::View<Kokkos::complex<float>\s*\*\s*, Kokkos::CudaSpace>>',
    )

    def test_cas_based_atomic(self, decoder: Decoder) -> None:
        """
        This test proves that it uses the atomic CAS-based atomic by looking for an instruction sequence pattern.
        """
        AtomicCAS(
            arch=self.arch,
            operation=AddComplex64(),
        ).match(cfg=ControlFlow.analyze(instructions=decoder.instructions))
