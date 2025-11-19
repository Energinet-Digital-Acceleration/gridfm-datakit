# Network Loading Feature

**Type:** Feature Diagram
**Last Updated:** 2025-11-10
**Related Files:**
- `gridfm_datakit/network.py:load_net_from_pp()`
- `gridfm_datakit/network.py:load_net_from_pglib()`
- `gridfm_datakit/network.py:load_net_from_pypowsybl()`
- `gridfm_datakit/network.py:load_net_from_file()`

## Purpose

Shows researchers how to load grids from multiple sources, with automatic bus reindexing ensuring consistent array operations across all network types.

## Diagram

```mermaid
flowchart TB
    subgraph "Front-Stage: Researcher Experience"
        User["👤 Researcher<br/>Needs: Load power grid<br/>from various sources"]
        ConfigSource["Config: network.source<br/>Options:<br/>- 'pandapower'<br/>- 'pglib'<br/>- 'pypowsybl'<br/>- 'file'"]
        Network["🔌 Loaded Network<br/>✓ Ready for simulation<br/>✓ Consistent bus indices<br/>✓ Validated structure"]
    end

    subgraph "Back-Stage: Loading Pipeline"
        direction TB

        subgraph "Source-Specific Loaders"
            PP["load_net_from_pp()<br/>Built-in pandapower grids<br/>Impact: Quick test cases"]
            PGLib["load_net_from_pglib()<br/>Industry-standard benchmarks<br/>Impact: Auto-downloads<br/>if missing"]
            PyPow["load_net_from_pypowsybl()<br/>PyPowSyBl network objects<br/>Impact: Interop with<br/>PyPowSyBl ecosystem"]
            File["load_net_from_file()<br/>MATPOWER .m files<br/>Impact: Custom grids"]
        end

        subgraph "Post-Load Processing"
            Reindex["Bus Reindexing<br/>Ensures 0...n-1 indices"]
            Validate["Structure Validation<br/>- All buses connected<br/>- Required fields present"]

            Critical["⚠️ Critical Operation<br/>Impact: Without reindexing,<br/>array operations fail"]
        end

        PP --> Reindex
        PGLib --> Reindex
        PyPow --> Reindex
        File --> Reindex

        Reindex --> Critical
        Critical --> Validate
    end

    User -->|"1. Specify in config"| ConfigSource

    ConfigSource -->|"source='pandapower'"| PP
    ConfigSource -->|"source='pglib'"| PGLib
    ConfigSource -->|"source='pypowsybl'"| PyPow
    ConfigSource -->|"source='file'"| File

    Validate -->|"2. Return validated network"| Network
    Network -->|"3. Use for generation"| User

    style User fill:#e1f5ff
    style Network fill:#90EE90
    style Critical fill:#ffcccc
    style Reindex fill:#fff4e1
```

## Key Insights

- **Source flexibility**: Researchers can use grids from multiple ecosystems without conversion
- **Auto-download**: PGLib grids fetched automatically (removes manual file management)
- **Reindexing criticality**: Ensures numpy array indexing works correctly (missing this causes hard-to-debug errors)
- **MATPOWER compatibility**: Legacy .m files supported for custom/proprietary grids
- **PyPowSyBl interop**: Can use networks from PyPowSyBl pipeline directly

## Change History

- **2025-11-10:** Initial network loading diagram created
