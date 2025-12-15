# PyPowSyBl Integration Work - MATPOWER edition

## Overview

This branch adds PyPowSyBl as an alternative power grid data source alongside pglib and pandapower.

## Running the IEEE 30 Cases

Two config files demonstrate the same IEEE 30-bus network from different sources:

```bash
# pglib source (OPF-modified data)
gridfm_datakit scripts/config/case30_ieee_mellson.yaml

# pypowsybl source (original IEEE data)
gridfm_datakit scripts/config/case30_ieee_pypowsybl.yaml
```

Outputs go to:
- `./data_out/pglib/case30_ieee/raw/`
- `./data_out/pypowsybl/case30_ieee/raw/`

## Technical Approach

### Problem
PyPowSyBl returns `pypowsybl.Network` objects, but the pipeline expects `pandapowerNet`.

### Solution: PyPowSyBl → MATPOWER → pandapower

```
pypowsybl.Network  →  .mat file  →  pandapowerNet
     (export)          (import)
```

Key steps in `load_net_from_pypowsybl()` (`gridfm_datakit/network.py`):
1. Call pypowsybl's `create_ieee30()` (or similar)
2. Export to temp MATPOWER file: `psy_net.save(path, format="MATPOWER")`
3. Import into pandapower: `pp.converter.from_mpc(path)`
4. Reindex buses to 0...n-1
5. Normalize for OPF compatibility

### Critical Fix: Transformer Impedance

The MATPOWER converter assigns `sn_mva=99999` when unspecified, which broke transformer impedance calculation:

```
vk_percent = x_pu × 100 × (sn_mva / baseMVA)
           = 0.2  × 100 × (99999 / 100)
           = 19,999%  ← caused OPF divergence
```

Fix in `_normalize_network_for_opf()`:
```python
if old_sn > 10000:
    new_sn = 100.0  # base MVA
    new_vk = old_vk * (new_sn / old_sn)  # scale proportionally
```

## Output Comparison: pglib vs pypowsybl

### pf_edge.csv (Admittance Matrix)

| Metric | Difference |
|--------|------------|
| G (conductance) | max 10⁻¹⁴, mean 10⁻¹⁶ |
| B (susceptance) | max 10⁻¹⁴, mean 10⁻¹⁶ |

**Result: Identical** - differences at floating-point precision level. Both sources represent the same electrical network topology.

### pf_node.csv (Power Flow Results)

| Metric | Max Diff | Mean Diff |
|--------|----------|-----------|
| Pd, Qd (loads) | 0.0 | 0.0 |
| Pg (gen power) | 103.9 MW | 6.6 MW |
| Vm (voltage mag) | 0.023 p.u. | 0.012 p.u. |
| Va (voltage angle) | 2.8° | 2.2° |

**Observations:**
- **Load demands identical**: Same input scenarios applied to both
- **Generator dispatch differs**: OPF finds different optimal operating points
  - pglib: Concentrates generation at bus 0 (167 MW)
  - pypowsybl: Distributes generation (64 MW bus 0, 100 MW bus 1)
- **Voltage profiles differ**: Consequence of different dispatch

### Why the Dispatch Differs

The difference is in **economic dispatch**, not physics. Both sources:
- Use the same network topology (identical admittance)
- Solve the same power flow equations
- Apply the same load scenarios

But pglib has OPF-modified cost curves that favor certain generators, while pypowsybl uses original IEEE reference data. Both are valid interpretations.

## Supported PyPowSyBl Grids

```python
_PYPOWSYBL_GRID_MAP = {
    "case9_ieee": ppsn.create_ieee9,
    "case14_ieee": ppsn.create_ieee14,
    "case30_ieee": ppsn.create_ieee30,
    "case57_ieee": ppsn.create_ieee57,
    "case118_ieee": ppsn.create_ieee118,
    "case300_ieee": ppsn.create_ieee300,
}
```
