---
title: 'ReProspect - A Framework for Reproducible prospecting of CUDA applications'
tags:
  - CUDA
  - reproducible
  - binary analysis
  - kernel profiling
  - API tracing
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
    corresponding: true
    email: maarten.arnst@uliege.be
affiliations:
  - name: University of Liège, Belgium
    index: 1
date: 01 Jan 2000
bibliography: paper.bib
---

# Summary

`ReProspect` is a Python framework designed to support reproducible prospecting of CUDA applications.
It builds on NVIDIA tools:
Nsight Systems for API tracing,
Nsight Compute for kernel profiling, and
CUDA binary tools for binary analysis.

`ReProspect` streamlines data collection and extraction using these tools,
and it complements them with new functionalities for a fully programmatic analysis of these data,
thus making it possible to encapsulate the entire prospecting workflow in a single Python script.

`ReProspect` provides a practical foundation for developing novel use cases of CUDA application prospecting.
It supports collaborative code development by enabling developers to share concise,
reproducible analyses that motivate design decisions and help reviewers grasp the impact of proposed changes.
It also enables new types of tests that go beyond traditional output-correctness validation in CI/CD pipelines
--- such as validating the presence of instruction patterns in binaries or confirming expected API call sequences for key library functionalities.
Additionally, `ReProspect` can act as a framework for structuring research artifacts that document analysis workflows,
enabling others to reproduce the work and build upon it more effectively.

# Statement of need

HPC software development strives to achieve performance and sustain it over time, while hardware and software evolve.
However, the modern programming landscape relies on complex software stacks and compiler toolchains.
Therefore, tools are needed to support developers in understanding, analysing and optimising the interaction of their code with other layers across the stack and ultimately with the hardware:
application runtime behaviour, individual kernel performance, and generated machine code.

The ability to carry out the analysis fully programmatically ensures reproducibility of the results by others while opening a range of new use cases that can be integrated in the development cycle.
For instance, whereas test suites traditionally check output correctness of public functionalities, they could also check API calls, hardware utilization, or code paths and instruction patterns in the generated machine code.

For the CUDA stack, NVIDIA provides a set of proprietary tools, guaranteed to be up-to-date with their software and hardware.
The runtime analysis tools Nsight Systems [REF] and Nsight Compute [REF] are designed for API tracing and kernel profiling.
They both provide a GUI for *ad hoc* exploratory understanding of the results, as well as a low-level Python API for accessing the raw data.
The CUDA binary tools [REF] provide command-line access to machine code (SASS or PTX [REF]) and other information embedded in the binaries.
However, these tools are designed to extract raw data and do not provide the infrastructure for effective programmatic analysis of these data.

Several well-established open-source analysis tools are already available.
The Lawrence Livermore National Laboratory (LLNL) provides a suite of tools.
Caliper [@boehme2016] can intercept CUDA API calls through the NVIDIA CUPTI library [@cupti2025].
It can interface with the Python package Hatchet [@bhatele2019] to organize results into a hierarchical data structure.
Thicket [@brink2023] adds kernel profiling support through Nsight Compute, with a primary focus on exploratory data analysis of multi-run performance experiments.
HPCToolkit [@zhou2021] is another comprehensive suite designed for large-scale parallel systems.
It includes CUDA API tracing through CUPTI and kernel profiling through PAPI [@terpstra2010], and it has binary analysis capabilities to aid with attributing performance data to calling contexts.
It has a visual interface for *ad hoc* exploration, and it can output raw performance data, such as through an interface to Hatchet, for programmatic analysis.
Score-P [@knupfer2012] integrates multiple performance analysis tools in a common infrastructure.
It can record CUDA API calls and GPU activities through CUPTI and provides standardized data formats.

As compared with these well-established tools, `ReProspect` targets different and new use cases: enabling concise, reproducible, script-driven analysis of individual units of functionality.
While tailored runtime analysis already proves useful for verifying library functionality, it requires the physical device, and runtime variability is an obstacle to adopting microbenchmarking as a verification method.
Notably, the machine code already contains information about the available code paths, and could itself be analysed and subjected to testing.
`ReProspect` thus goes beyond runtime analysis and introduces new binary analysis functionalities to inspect machine code for expected instruction sequence patterns.

# Key features

Key features.

# Case studies

A few nice case studies.

# Code availability

`ReProspect` is available on [GitHub](https://github.com/uliegecsm/reprospect).

# Acknowledgements

This work is supported by FRIA.

# References
