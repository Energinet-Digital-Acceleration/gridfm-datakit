# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gridfm-datakit is a Python library for generating power flow datasets to train machine learning and foundation models. It generates synthetic power grid data through:
- Load scenario generation (using aggregated load profiles or PowerGraph)
- Topology perturbations (N-k contingency analysis, random perturbations)
- Generation cost perturbations
- Network parameter perturbations

The library uses pandapower for power flow simulations and supports both sequential and distributed multiprocessing execution.

## Key Commands

### Installation
```bash
pip install -e .                    # Install in editable mode
pip install -e ".[dev]"             # Install with development dependencies
pip install -e ".[test]"            # Install with test dependencies
```

### Testing
```bash
pytest tests/                       # Run all tests
pytest tests/test_generate.py       # Run specific test file
pytest tests/ -v                    # Run with verbose output
pytest tests/ --config scripts/config/default.yaml  # Run with specific config
```

### CLI Usage
```bash
gridfm_datakit path/to/config.yaml  # Generate data from config file
```

### Interactive Usage
Open `scripts/interactive_interface.ipynb` or use:
```python
from gridfm_datakit.interactive import interactive_interface
interactive_interface()
```

## Architecture

### Core Data Generation Pipeline

1. **Network Loading** (`network.py`):
   - Loads power grids from three sources: pandapower, pglib, or matpower files
   - All networks are reindexed to ensure continuous bus indices
   - PGLib networks are automatically downloaded if not locally available

2. **Load Scenario Generation** (`perturbations/load_perturbation.py`):
   - Abstract base class `LoadScenarioGeneratorBase` with two implementations:
     - `AggregateLoadProfileGenerator`: Uses aggregated load profiles with noise
     - `PowerGraphGenerator`: Uses PowerGraph method
   - Returns 3D numpy array of shape (n_loads, n_scenarios, 2) for p_mw and q_mvar

3. **Topology Perturbation** (`perturbations/topology_perturbation.py`):
   - Abstract base class `TopologyGenerator` with implementations:
     - `NoPerturbationGenerator`: No modifications
     - `NMinusKGenerator`: N-k contingency analysis (all combinations)
     - `RandomTopologyGenerator`: Random component removal
   - Only yields topologies that are feasible (no unsupplied buses)
   - Can perturb lines, transformers, generators, and static generators

4. **Power Flow Solving** (`process/solvers.py`):
   - `run_opf()`: Runs optimal power flow with validation checks
   - `run_pf()`: Runs regular power flow
   - Both add bus index and type information to result dataframes
   - Extensive assertions validate power bounds and balance

5. **Data Processing** (`generate.py`):
   - Two main entry points:
     - `generate_power_flow_data()`: Sequential processing
     - `generate_power_flow_data_distributed()`: Multiprocessing with chunked saves
   - Modular design with private functions:
     - `_setup_environment()`: Creates output directories and file paths
     - `_prepare_network_and_scenarios()`: Loads network and generates scenarios
     - `_save_generated_data()`: Saves all output files
   - Supports two modes: "pf" (power flow) and "contingency"

### Configuration System

Uses `NestedNamespace` (`utils/param_handler.py`) to convert YAML configs into nested objects with dot notation access. Config structure:
- `network`: Grid source and name
- `load`: Load scenario generation parameters
- `topology_perturbation`: Topology perturbation settings
- `generation_perturbation`: Generator cost perturbation settings
- `admittance_perturbation`: Network parameter perturbation settings
- `settings`: Execution settings (num_processes, data_dir, mode)

### Output Files

All outputs are saved to `{data_dir}/{network_name}/raw/`:
- `pf_node.csv`: Node (bus) data for each power flow case
- `pf_edge.csv`: Edge (branch) data for each power flow case
- `branch_idx_removed.csv`: Indices of removed branches in topology perturbations
- `edge_params.csv`: Branch admittance matrix and rate limits for base topology
- `bus_params.csv`: Bus voltage limits and base voltages
- `scenarios_{generator}.csv`: Element-level load profiles
- `scenarios_{generator}.html`: Load profile plots
- `stats.csv` and `stats_plot.html`: Statistics about generated data
- `tqdm.log`, `error.log`, `args.log`: Logging files

## Important Design Patterns

### Abstract Base Classes
The codebase uses ABC for extensibility:
- `LoadScenarioGeneratorBase`: Load scenario generators
- `TopologyGenerator`: Topology perturbation generators
- Similar patterns for generation and admittance perturbations

### Multiprocessing Pattern
Distributed generation uses:
- Manager Queue for progress tracking
- Chunked processing (large_chunk_size) to limit memory usage
- Intermediate saves after each chunk
- Process-safe error logging

### Network Reindexing
All loaded networks have buses reindexed to 0...n-1 for consistent array indexing. This is critical for correct data generation.

## Testing

Tests use pytest fixtures:
- `conf`: Loads default config from `tests/config/default.yaml`
- `cleanup_generated_files`: Removes test output directories
- Tests validate both sequential and distributed generation
- Tests cover both "pf" and "contingency" modes

## Known Constraints

- `net.sgen["scaling"]` must be 1.0 (scaling factors >1 not yet supported)
- k > 1 in N-k contingency may be very slow (generates combinatorial topologies)
- Some pglib grids may not converge and are filtered out
- Python 3.10-3.12 supported (pandas/pandapower compatibility)

## Development Notes

- The codebase is designed for high-volume data generation (1M+ samples)
- Memory management is critical in distributed mode
- Network validation is extensive (power bounds, balance checks, etc.)
- Error logging is per-scenario to avoid failing entire runs
