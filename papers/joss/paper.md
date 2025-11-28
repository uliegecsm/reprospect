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

`ReProspect` is a `Python` framework designed to support reproducible prospecting of CUDA applications.
It builds upon NVIDIA tools:
Nsight Systems for API tracing,
Nsight Compute for kernel profiling, and
CUDA binary tools for binary analysis.

`ReProspect` streamlines data collection and extraction using these tools,
and complements them with new functionalities for a fully programmatic analysis of these data,
making it possible to encapsulate the entire prospecting workflow in a single Python script.

`ReProspect` provides a practical foundation for developing novel use cases of CUDA application prospecting.
It supports collaborative code development by enabling developers to share concise,
reproducible analyses that motivate design decisions and help reviewers grasp the impact of proposed changes.
It also enables new types of tests that go beyond traditional output-correctness validation in CI/CD pipelines
--- such as validating the presence of instruction patterns in binaries or confirming expected API call sequences for key library functionalities.
Additionally, `ReProspect` can act as a framework for structuring research artifacts that document analysis workflows,
enabling others to reproduce the work and build upon it more effectively.

# Statement of need

The statement of needs comes here.

# Key features

Key features.

# Case studies

A few nice case studies.

# Code availability

`ReProspect` is available on [GitHub](https://github.com/uliegecsm/reprospect).

# Acknowledgements

This work is supported by FRIA.

# References
