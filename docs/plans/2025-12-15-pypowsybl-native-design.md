# Native pypowsybl Integration Design

## Overview

Add pypowsybl as a native data source using its own AC load flow solver, without converting to pandapower format.

## Architecture

Two parallel code paths based on `network.source`:

```
source="pglib" | "pandapower" | "file":
  → pandapowerNet → run_opf/run_pf → pf_post_processing → output

source="pypowsybl":
  → pypowsybl.Network → run_ac() → pypowsybl_post_processing → output (same format)
```

## Data Mapping

### pf_node.csv

| Output Column | pypowsybl Source | Notes |
|---------------|------------------|-------|
| bus | Bus index (0, 1, 2...) | Map string IDs → integer |
| Pd | `get_loads()['p']` | Aggregate by bus |
| Qd | `get_loads()['q']` | Aggregate by bus |
| Pg | `get_generators()['p']` | Negate (load convention) |
| Qg | `get_generators()['q']` | Negate |
| Vm | `get_buses()['v_mag'] / nominal_v` | kV → p.u. |
| Va | `get_buses()['v_angle']` | Already in degrees |
| PQ/PV/REF | Derive from generator connections | REF = slack bus |

### pf_edge.csv (Y-bus matrix)

Build from `get_lines()` and `get_2_windings_transformers()`:

```python
y_series = 1 / (r + jx)
y_shunt = j * (b1 + b2) / 2
Y[i,i] += y_series + y_shunt
Y[j,j] += y_series + y_shunt
Y[i,j] -= y_series
Y[j,i] -= y_series
```

## Load Scenario Application

```python
load_df = pd.DataFrame({
    'id': load_ids,
    'p0': scenarios[:, scenario_index, 0],
    'q0': scenarios[:, scenario_index, 1]
})
network.update_loads(load_df)
```

## Scope & Limitations

Initial implementation supports:
- AC Power Flow only (no OPF)
- `mode: "pf"` only
- `topology_perturbation.type: "none"`

Not supported initially:
- OPF (not needed for pypowsybl use case)
- Topology perturbation
- Generation/admittance perturbation

## Files to Modify

| File | Changes |
|------|---------|
| `network.py` | Simplify `load_net_from_pypowsybl()` to return native Network |
| `process/solvers.py` | Add `run_pf_pypowsybl()` |
| `process/process_network.py` | Add `pypowsybl_post_processing()`, `get_adjacency_list_pypowsybl()`, `process_scenario_pypowsybl()` |
| `generate.py` | Branch logic based on source type |

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
