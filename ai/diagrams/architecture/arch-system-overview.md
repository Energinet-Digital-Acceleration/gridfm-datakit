# System Architecture Overview

**Type:** Architecture Diagram
**Last Updated:** 2025-11-10
**Related Files:**
- `gridfm_datakit/generate.py`
- `gridfm_datakit/network.py`
- `gridfm_datakit/process/solvers.py`
- `gridfm_datakit/perturbations/*.py`

## Purpose

Shows ML researchers/data scientists how synthetic power grid training data flows from network loading through perturbations to validated datasets, enabling foundation model training.

## Diagram

```mermaid
flowchart TB
    subgraph "Front-Stage: ML Training Data Pipeline"
        User["👤 ML Researcher<br/>Needs: Training datasets<br/>for power grid models"]
        Config["📄 YAML Config<br/>Defines grid, scenarios,<br/>perturbation strategies"]
        Output["📊 Training Data<br/>✓ Validated power flow cases<br/>✓ Topology variations<br/>✓ Load scenarios<br/>Impact: Ready for model training"]
    end

    subgraph "Back-Stage: Data Generation Engine"
        direction TB

        subgraph "1. Network Loading"
            NetLoader["Network Loader<br/>network.py"]
            Sources["Sources:<br/>- pandapower<br/>- PGLib<br/>- PyPowSyBl<br/>- MATPOWER files"]
            Reindex["Bus Reindexing<br/>Impact: Ensures consistent<br/>0...n-1 array indexing"]
        end

        subgraph "2. Perturbation Pipeline"
            LoadPert["Load Scenarios<br/>load_perturbation.py<br/>Impact: Realistic load variations"]
            TopoPert["Topology Perturbation<br/>topology_perturbation.py<br/>Impact: N-k contingency analysis"]
            GenPert["Generation Cost<br/>generator_perturbation.py"]
            AdmPert["Network Parameters<br/>admittance_perturbation.py"]
        end

        subgraph "3. Power Flow Solving & Validation"
            Solver["OPF/PF Solver<br/>solvers.py<br/>Impact: Ensures physically<br/>valid solutions"]
            Validation["Validation Checks:<br/>- Power bounds<br/>- Balance equations<br/>- Convergence<br/>Impact: Filters bad data"]
        end

        subgraph "4. Processing Strategy"
            Sequential["Sequential Mode<br/>generate_power_flow_data()"]
            Distributed["Distributed Mode<br/>Multiprocessing with<br/>chunked saves<br/>Impact: Handles 1M+ samples"]
        end
    end

    User -->|"1. Define requirements"| Config
    Config -->|"2. Initialize"| NetLoader
    NetLoader --> Sources
    Sources --> Reindex

    Reindex -->|"3. Apply"| LoadPert
    Reindex -->|"3. Apply"| TopoPert
    LoadPert --> GenPert
    TopoPert --> GenPert
    GenPert --> AdmPert

    AdmPert -->|"4. Solve"| Solver
    Solver --> Validation

    Validation -->|"5. Choose strategy"| Sequential
    Validation -->|"5. Choose strategy"| Distributed

    Sequential -->|"6. Write validated data"| Output
    Distributed -->|"6. Write validated data"| Output
    Output -->|"7. Train models"| User

    style User fill:#e1f5ff
    style Output fill:#90EE90
    style Validation fill:#fff4e1
    style Distributed fill:#fff4e1
```

## Key Insights

- **ML researcher value**: Single command generates thousands of validated power grid scenarios for training
- **Quality guarantee**: Multi-stage validation ensures physically plausible data (no training on impossible states)
- **Scale enabler**: Distributed processing with chunking handles production-scale datasets (1M+ samples) without memory issues
- **Flexibility**: Plugin architecture (ABC pattern) allows researchers to add custom perturbation strategies
- **Data integrity**: Bus reindexing ensures array operations work correctly across all network types

## Change History

- **2025-11-10:** Initial architecture diagram created
