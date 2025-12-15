# PyPowSyBl Native Integration

## Overview

This branch adds PyPowSyBl as a native power grid data source using its own AC load flow solver, without converting to pandapower format.

## Running the IEEE 30 Cases

```bash
# pglib source (OPF via pandapower)
gridfm_datakit scripts/config/case30_ieee_mellson.yaml

# pypowsybl source (native AC power flow)
gridfm_datakit scripts/config/case30_ieee_pypowsybl.yaml
```

## Technical Approach

### Architecture: Hybrid Path

Two parallel code paths based on `network.source`:

```
source="pglib" | "pandapower" | "file":
  → pandapowerNet → run_opf/run_pf → pf_post_processing → output

source="pypowsybl":
  → PyPowSyBlNetwork → run_ac() → pypowsybl_post_processing → output (same format)
```

### Why Native Instead of MATPOWER Conversion?

The previous approach (`pypowsybl-matpower` branch) converted pypowsybl networks to pandapower via MATPOWER format:

```
pypowsybl.Network → .mat file → pandapowerNet
```

**Problems with MATPOWER conversion:**
1. MATPOWER converter assigns `sn_mva=99999` for unspecified transformer ratings
2. This caused transformer impedance (`vk_percent`) to be 1000x too high
3. Required manual fixes in `_normalize_network_for_opf()`
4. Conversion chain loses some network metadata

**Native approach advantages:**
1. No format conversion - use pypowsybl directly
2. Uses pypowsybl's own `loadflow.run_ac()` solver
3. Simpler, more maintainable code
4. Preserves original IEEE network parameters

### Key Components

| File | Function | Purpose |
|------|----------|---------|
| `network.py` | `load_net_from_pypowsybl()` | Returns `PyPowSyBlNetwork` container |
| `network.py` | `PyPowSyBlNetwork` | Dataclass with network + metadata |
| `solvers.py` | `run_pf_pypowsybl()` | Wrapper for pypowsybl AC load flow |
| `process_network.py` | `pypowsybl_post_processing()` | Extracts pf_node.csv format |
| `process_network.py` | `get_adjacency_list_pypowsybl()` | Builds Y-bus matrix |
| `generate.py` | `generate_power_flow_data()` | Branches based on source type |

### Data Mapping

**pf_node.csv:**
- `Vm`: `get_buses()['v_mag'] / nominal_v` → p.u.
- `Va`: `get_buses()['v_angle']` → degrees
- `Pg/Qg`: `-get_generators()['p'/'q']` (negate due to load convention)
- `Pd/Qd`: `get_loads()['p'/'q']`
- `PQ/PV/REF`: Derived from generator connections

**pf_edge.csv (Y-bus):**
- Built from `get_lines()` and `get_2_windings_transformers()`
- Line admittance: `y = 1/(r + jx)`, shunt: `jb`
- Transformer model includes turns ratio

## Limitations

| Feature | pypowsybl | pandapower |
|---------|-----------|------------|
| AC Power Flow | ✓ | ✓ |
| OPF | ✗ | ✓ |
| Topology perturbation | ✗ | ✓ |
| Generation perturbation | ✗ | ✓ |
| Admittance perturbation | ✗ | ✓ |
| Contingency mode | ✗ | ✓ |

**pypowsybl path only supports:**
- `mode: "pf"`
- `topology_perturbation.type: "none"`

## Supported Grids

```python
_PYPOWSYBL_GRID_MAP = {
    "case9_ieee": create_ieee9,
    "case14_ieee": create_ieee14,
    "case30_ieee": create_ieee30,
    "case57_ieee": create_ieee57,
    "case118_ieee": create_ieee118,
    "case300_ieee": create_ieee300,
}
```

## Branch History

- `pypowsybl-matpower`: MATPOWER conversion approach (preserved)
- `pypowsybl`: Native pypowsybl solver approach (this branch)
