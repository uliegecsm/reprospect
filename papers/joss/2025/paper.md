---
title: 'ReProspect - A framework for reproducible prospecting of CUDA applications'
tags:
  - CUDA
  - reproducible
  - API tracing
  - kernel profiling
  - binary analysis
  - HPC
  - Kokkos
authors:
  - name: Romin Tomasetti
    orcid: 0009-0006-2839-7072
    affiliation: 1
    equal-contrib: true
    corresponding: true
    email: romin.tomasetti@uliege.be
  - name: Maarten Arnst
    orcid: 0000-0003-4993-0527
    affiliation: 1
    equal-contrib: true
    email: maarten.arnst@uliege.be
affiliations:
  - name: University of Liège, Belgium
    index: 1
date: 31 Dec 2025
bibliography: paper.bib
---

# Summary

`ReProspect` is a Python framework designed to support reproducible prospecting of CUDA code---that is, the systematic analysis of CUDA-based libraries and software components through API tracing, kernel profiling, and binary analysis.

`ReProspect` builds on NVIDIA tools:
Nsight Systems,
Nsight Compute, and
the CUDA binary utilities.
It streamlines data collection and extraction using these tools,
and it complements them with new functionalities for a fully programmatic analysis of these data,
thus making it possible to encapsulate the entire prospecting analysis in a single Python script.

`ReProspect` provides a practical foundation for developing novel use cases of CUDA code prospecting.
It supports collaborative code development by enabling developers to share concise,
reproducible analyses that motivate design decisions and help reviewers grasp the impact of proposed changes.
It also enables new types of tests that go beyond traditional output-correctness validation in CI/CD pipelines,
such as validating the presence of instruction patterns in binaries or confirming expected API call sequences for key library functionalities.
Additionally, `ReProspect` can act as a framework for structuring research artifacts and documenting analyses,
enabling others to reproduce the work and build upon it more effectively.

# Statement of need

HPC software development strives to achieve performance and sustain it over time, while hardware and software evolve.
However, the modern programming landscape relies on complex software stacks and compiler toolchains.
Therefore, tools are needed to analyse the interaction of the code with other layers across the stack and ultimately with the hardware.

The ability to carry out the analysis fully programmatically ensures reproducibility of the results by others while opening a range of new use cases that can be integrated in the development cycle.
For instance, whereas test suites traditionally check output correctness of public functionalities,
they could also verify application runtime events, kernel performance, or generated machine code.

For the CUDA stack, NVIDIA provides a set of proprietary tools guaranteed to be up-to-date with their software and hardware.
The runtime analysis tools Nsight Systems [@nsys2025] and Nsight Compute [@ncu2025] are designed for API tracing and kernel profiling, respectively.
They both provide a GUI for exploring the results, as well as a low-level Python API for accessing the raw data.
The CUDA binary utilities [@binary2025] provide command-line access to machine code (SASS or PTX [@ptx2025]) and other information embedded in the binaries.
However, while these tools allow raw data to be extracted, they
do not themselves provide the infrastructure for effective programmatic analysis.

# State of the field

Several well-established open-source tools are already available.
Caliper [@boehme2016] can intercept CUDA API calls through the NVIDIA CUPTI library [@cupti2025].
It can interface with the Python package Hatchet [@bhatele2019] to organize results into a hierarchical data structure.
Thicket [@brink2023] adds kernel profiling support through Nsight Compute, with a primary focus on exploratory data analysis of multi-run performance experiments.
HPCToolkit [@zhou2021] is another comprehensive suite designed for large-scale parallel systems.
It includes CUDA API tracing through CUPTI and kernel profiling through PAPI [@terpstra2010], and it has binary analysis capabilities to attribute performance data to calling contexts.
It has a visual interface, and it can output raw performance data for programmatic analysis,
*e.g.* using Hatchet.
Score-P [@knupfer2012] integrates multiple performance analysis tools in a common infrastructure.
It can record CUDA API calls and GPU activities through CUPTI and provides standardized data formats.

Although script-driven runtime analysis is also possible with these well-established tools,
developing `ReProspect` as an independent package enables a design optimised for our use cases:
concise, reproducible, low-overhead, easy-to-adopt, script-driven analysis of individual units of functionality.
Beyond runtime analysis, `ReProspect` introduces new binary analysis functionalities to inspect machine code for expected instruction sequence patterns.
To the best of our knowledge, these functionalities are not covered by existing tools.

# Software design

`ReProspect` is organized into three main components:
API tracing, kernel profiling, and binary analysis (\autoref{fig:overview}).

Each component design stems from the identification, organisation and implementation of recurring functionalities,
yielding modules and classes from which concise analysis scripts can be written.
This includes launching the underlying analysis tool,
collecting the output into Python data structures,
and performing the subsequent analysis.

![Overview of `ReProspect`.\label{fig:overview}](overview.svg){width="12cm"}

## API tracing and kernel profiling

The `ReProspect` `Command` and `Session` classes streamline launching
Nsight Systems and Nsight Compute to collect
a focused set of metrics most relevant for the analysis.
The collected data are gathered in a `Report`, queryable by NVTX range annotations [@nvtx],
readily amenable to test assertions.

To avoid unnecessary re-runs,
`ReProspect` provides a `Cacher` that can serve the `Report` from a database.

## Binary analysis

`ReProspect` provides a set of tools for extracting and analysing the content of CUDA binaries.

The `CuObjDump` and `NVDisasm` classes drive and parse the output of the underlying CUDA binary utilities
to retrieve the SASS code and resource usage of kernels (*e.g.* registers, constant memory).

The `ELF` class decodes ELF-formatted sections to extract complementary information,
including the symbol table, toolchain metadata (from the *.note.nv.tkinfo* section), and
kernel attributes such as launch bounds (from the *.nv.info.\<kernel\>* section) [@hayes-2019-decoding-cubin].

Beyond data extraction, `ReProspect` provides an extensible framework for
inspecting machine code for expected instruction patterns.
This framework is structured as a hierarchy of matchers.
At the lowest levels, matchers analyse instructions and their components (opcodes, modifiers, and operands).
These matchers can be composed into instruction sequence matchers,
enabling the identification of more intricate patterns.

One of the key challenges with SASS matching is the evolution of the CUDA instruction set and compiler toolchains.
`ReProspect` addresses this challenge by abstracting away architecture- and toolchain-specific details at each level of the matcher hierarchy.
For example, `AddressMatcher` matches a memory address operand, adjusting the expected address format for the target architecture.
Then, at the instruction level, `LoadMatcher` matches loads from memory; it uses `AddressMatcher` for the memory address and thus needs only to adjust the opcode and modifiers for the target architecture.
By extending this hierarchical design to instruction sequence and basic block matchers,
`ReProspect` enables robust, composable matching across CUDA architectures and compiler toolchains.

The following snippet illustrates how matchers can be composed to
assert the presence or absence of a 16-bit floating-point code path, *e.g.* in the SASS codes in \autoref{table:hfmax}:
```python
arch = NVIDIAArch(...)
instructions = Decoder(...)
cfg = ControlFlow.analyze(instructions)

matcher_ldg = instructions_contain(instructions_are(
    LoadGlobalMatcher(arch, size=16, extend='U', readonly=False),
    LoadGlobalMatcher(arch, size=16, extend='U', readonly=False),
))
blk, matched_ldg = BasicBlockMatcher(matcher_ldg).match(cfg=cfg)

matcher_hmnmx2 = instructions_contain(instruction_is(
    Fp16MinMaxMatcher(pmax=True))
    .with_operand(index=1, operand=f'{matched_ldg[0].operands[0]}.H0_H0')
    .with_operand(index=2, operand=f'{matched_ldg[1].operands[0]}.H0_H0')
    .with_operand(index=0, operand=RegisterMatcher(special=False))
)
matched_hmnmx2 = matcher_hmnmx2.match(blk.instructions[matcher_ldg.next_index:])
```

+--------------------------------------+----------------------------------+
| ```c                                 | ```c                             |
| __hmax(const __half, const __half)   | fmax(const float, const float)   |
| ```                                  | ```                              |
+======================================+==================================+
| ```sass                              | ```sass                          |
| LDG.E.U16 R2, desc[UR6][R2.64]       | LDG.E.U16 R2, desc[UR6][R2.64]   |
| LDG.E.U16 R5, desc[UR6][R4.64]       | LDG.E.U16 R4, desc[UR6][R4.64]   |
| ...                                  | ...                              |
| HMNMX2 R5, R2.H0_H0, R5.H0_H0, !PT   | HADD2.F32 R6, -RZ, R2.H0_H0      |
| ...                                  | HADD2.F32 R7, -RZ, R4.H0_H0      |
|                                      | FMNMX R6, R6, R7, !PT            |
|                                      | F2FP.F16.F32.PACK_AB R3, RZ, R6  |
|                                      | ...                              |
| STG.E.U16 desc[UR6][R6.64], R5       | STG.E.U16 desc[UR6][R6.64], R3   |
| ```                                  | ```                              |
+--------------------------------------+----------------------------------+
Table: Comparison of the SASS code generated for the `sm_100` architecture
       for the 16-bit `__half` (left) *vs* 32-bit `float` (right) maximum function.
       \label{table:hfmax}

# Research impact statement

`ReProspect` has evolved from conducting analyses in support of contributions
to the open-source Kokkos library [@ctrott-2022] and the development of an in-house finite element
code built on top of the open-source Trilinos library [@mayr-2025] [@arnst-24] [@tomasetti-24].

The [`examples` directory](https://github.com/uliegecsm/reprospect/blob/54a95f066cbf350a54305457159aafdd751f1b18/examples/)
contains several case studies inspired by these research efforts.

## `Kokkos::View` allocation

CUDA API tracing provides insight into microbenchmarking results assessing the behavior of `Kokkos::View` allocation.

See [online example](https://github.com/uliegecsm/reprospect/blob/54a95f066cbf350a54305457159aafdd751f1b18/examples/kokkos/view/example_allocation_tracing.py)
and [Kokkos issue](https://github.com/kokkos/kokkos/issues/8441).

## Impact of `Kokkos::complex` alignment

Our in-house finite element code uses complex arithmetic for frequency-domain electromagnetism simulations.
This example combines kernel profiling and SASS analysis to assess how aligning `Kokkos::complex<double>`
to 8 or 16 bytes impacts memory instructions and traffic.

See [online example](https://github.com/uliegecsm/reprospect/blob/54a95f066cbf350a54305457159aafdd751f1b18/examples/kokkos/complex/example_alignment.py).

## Dynamic dispatch for virtual functions on device

This example uses `ReProspect` to create a research artifact that reproduces the dynamic dispatch instruction pattern
identified by [@zhang-2021].

See [online example](https://github.com/uliegecsm/reprospect/blob/54a95f066cbf350a54305457159aafdd751f1b18/examples/cuda/virtual_functions/example_dispatch.py).

## Atomics with `desul`

Kokkos provides extended atomic support through the `desul` library [@ctrott-2022],
which maps atomic operations to one of several methods with varying performance.
The choice of the method depends on intricate logic, and must be tested.
A micro-benchmarking approach is feasible, but requires the physical device
and suffers from runtime variability.
Yet, the machine code already contains information about the selected code paths.
This case study demonstrates how to verify which method is chosen
by matching an instruction sequence pattern.

See [online example](https://github.com/uliegecsm/reprospect/blob/54a95f066cbf350a54305457159aafdd751f1b18/examples/kokkos/atomic/desul.py).

# Code availability

`ReProspect` is available under the `LGPL-3.0` license on [GitHub](https://github.com/uliegecsm/reprospect).

# Acknowledgements

This work was supported by the Fonds de la Recherche Scientifique (F.R.S.-FNRS, Belgium) through a
Research Fellowship.

# AI usage disclosure

No AI was used for the design of the code.
Alongside traditional tools such as `pylint` and `mypy`, `Claude` and `CoPilot` were used as assistants
to improve implementation details of individual functions.
AI helped improve the clarity of the manuscript.

# References
