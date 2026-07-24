"""
At first glance, examining generated SASS code may appear to be an esoteric task suited only to expert lab researchers — let alone testing it.

Yet, studying SASS — and assembly code in general — offers valuable insights.
Indeed, modern HPC code bases rely on complex software stacks and compiler toolchains.
While code correctness is often checked through regression testing,
reaching and sustaining optimal performance as software and hardware evolve requires additional effort.
This is usually achieved through verification of compile flags and *ad hoc* profiling and benchmarking.
However, beyond runtime analysis, the SASS code already contains information about the available code paths and could itself be incorporated into testing.
Still, the barrier to entry for meaningful SASS analysis is high: results can vary dramatically with compiler versions, optimization flags, and target architectures.

`ReProspect` provides a hierarchy of SASS instruction matchers that capture the components of an instruction (*opcodes*, *modifiers* and *operands*).
Under the hood, they generate complex regular expression (*regex*) patterns.
To accommodate for the evolving CUDA Instruction Set,
some of these matchers take the target architecture as a parameter and adjust the regex patterns accordingly.
In this way, `ReProspect` helps developers write assertions about expected instructions,
while reducing the need to track low-level details of the evolving CUDA instruction set.

.. doctest::

    >>> from reprospect.tools.architecture import NVIDIAArch
    >>> from reprospect.testing.binaries.sass.instruction import LoadGlobalMatcher
    >>> LoadGlobalMatcher(arch = NVIDIAArch.from_str('VOLTA70')).match(inst = 'LDG.E.SYS R15, [R8+0x10]')
    InstructionMatch(opcode='LDG', modifiers=('E', 'SYS'), operands=('R15', '[R8+0x10]'), predicate=None, additional={'address': ['[R8+0x10]']})
    >>> LoadGlobalMatcher(arch = NVIDIAArch.from_str('BLACKWELL120'), size = 128, readonly = True).match(inst = 'LDG.E.128.CONSTANT R2, desc[UR15][R6.64+0x12]')
    InstructionMatch(opcode='LDG', modifiers=('E', '128', 'CONSTANT'), operands=('R2', 'desc[UR15][R6.64+0x12]'), predicate=None, additional={'address': ['desc[UR15][R6.64+0x12]']})

References:

* https://github.com/Usibre/asmregex
* :cite:`yan-optimizing-winograd-2020`
* https://github.com/0xD0GF00D/DocumentSASS
* https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuInsParser.py#L9C75-L9C83
"""

from .address import (
    AddressMatcher,
    GenericOrGlobalAddressMatch,
    LocalAddressMatch,
    SharedAddressMatch,
    StrideModifier,
)
from .atomic import AtomicMatcher, ReductionMatcher, ThreadScope
from .branch import BranchMatcher
from .constant import Constant, ConstantMatch, ConstantMatcher
from .floating import (
    Fp32AddMatcher,
    Fp64AddMatcher,
)
from .half import (
    Fp16AddMatcher,
    Fp16FusedMulAddMatcher,
    Fp16MinMaxMatcher,
    Fp16MulMatcher,
)
from .immediate import Immediate
from .instruction import (
    AnyMatcher,
    ArchitectureAndVersionAwarePatternMatcher,
    ArchitectureAwarePatternMatcher,
    InstructionMatch,
    InstructionMatcher,
    OpcodeModsMatcher,
    OpcodeModsWithOperandsMatcher,
    PatternMatcher,
    Predicate,
    ZeroOrOne,
)
from .integer import (
    IntAdd3Matcher,
    IntAddMatcher,
    LEAMatcher,
)
from .load import (
    LoadConstantMatcher,
    LoadGlobalMatcher,
    LoadMatcher,
)
from .memory import MemorySpace
from .operand import MathModifier, Operand
from .pattern import PatternBuilder
from .register import Register, RegisterMatch, RegisterMatcher
from .store import StoreGlobalMatcher, StoreMatcher
from .validate import (
    ModifierValidator,
    OperandMatcher,
    OperandsValidator,
    OperandValidator,
)

__all__ = (
    'AddressMatcher',
    'AnyMatcher',
    'ArchitectureAndVersionAwarePatternMatcher',
    'ArchitectureAwarePatternMatcher',
    'AtomicMatcher',
    'BranchMatcher',
    'Constant',
    'ConstantMatch',
    'ConstantMatcher',
    'Fp16AddMatcher',
    'Fp16FusedMulAddMatcher',
    'Fp16MinMaxMatcher',
    'Fp16MulMatcher',
    'Fp32AddMatcher',
    'Fp64AddMatcher',
    'GenericOrGlobalAddressMatch',
    'Immediate',
    'InstructionMatch',
    'InstructionMatcher',
    'IntAdd3Matcher',
    'IntAddMatcher',
    'LEAMatcher',
    'LoadConstantMatcher',
    'LoadGlobalMatcher',
    'LoadMatcher',
    'LocalAddressMatch',
    'MathModifier',
    'MemorySpace',
    'ModifierValidator',
    'OpcodeModsMatcher',
    'OpcodeModsWithOperandsMatcher',
    'Operand',
    'OperandMatcher',
    'OperandValidator',
    'OperandsValidator',
    'PatternBuilder',
    'PatternMatcher',
    'Predicate',
    'ReductionMatcher',
    'Register',
    'RegisterMatch',
    'RegisterMatcher',
    'SharedAddressMatch',
    'StoreGlobalMatcher',
    'StoreMatcher',
    'StrideModifier',
    'ThreadScope',
    'ZeroOrOne',
)
