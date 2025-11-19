# Unified Impact Diagrams Index

This directory contains all diagrams for gridfm-datakit, following Diagram Driven Development (DDD) methodology.

## Architecture Overview

- [System Architecture](architecture/arch-system-overview.md) - Complete data generation pipeline from network loading to validated datasets

## User Journeys

- [CLI to Dataset Journey](journeys/sequence-cli-to-dataset.md) - Complete user experience from CLI command to training data, including error handling and progress feedback

## Features

- [Data Generation Pipeline](features/feature-data-generation-pipeline.md) - Sequential vs distributed processing modes with performance trade-offs
- [Perturbation Strategies](features/feature-perturbation-strategies.md) - Load, topology, generation, and network parameter perturbations for dataset diversity
- [Network Loading](features/feature-network-loading.md) - Multi-source grid loading (pandapower, PGLib, PyPowSyBl, MATPOWER) with automatic reindexing

## Test Coverage

_(No test diagrams created yet)_

## Refactoring Plans

_(No refactoring diagrams created yet)_

## Last Updated

2025-11-10 - Initial diagram set created covering core architecture, user journeys, and key features
