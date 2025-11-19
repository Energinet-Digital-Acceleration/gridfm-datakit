# Data Generation Pipeline Feature

**Type:** Feature Diagram
**Last Updated:** 2025-11-10
**Related Files:**
- `gridfm_datakit/generate.py:generate_power_flow_data()`
- `gridfm_datakit/generate.py:generate_power_flow_data_distributed()`
- `gridfm_datakit/process/process_network.py`

## Purpose

Shows researchers how their config file transforms into training data through the sequential and distributed generation modes, highlighting performance trade-offs.

## Diagram

```mermaid
sequenceDiagram
    actor User as 👤 Researcher
    participant CLI as CLI Entry
    participant Setup as _setup_environment()
    participant Prepare as _prepare_network<br/>_and_scenarios()
    participant Process as Processing Loop
    participant Save as _save_generated_data()
    participant Output as 📊 Dataset Files

    Note over User,Output: Impact: Transforms config → validated ML training data

    User->>CLI: gridfm_datakit config.yaml
    Note right of User: Specifies: grid, load scenarios,<br/>topology perturbations, mode

    CLI->>Setup: Initialize
    Setup->>Setup: Create output dirs
    Setup->>Setup: Setup logging (tqdm, error, args)
    Setup-->>CLI: base_path, file_paths
    Note right of Setup: Impact: Organized output structure<br/>for reproducibility

    CLI->>Prepare: Load network & generate scenarios
    Prepare->>Prepare: Load network (PP/PGLib/PyPowSyBl)
    Prepare->>Prepare: Reindex buses (0...n-1)
    Note right of Prepare: Critical: Ensures consistent<br/>array indexing
    Prepare->>Prepare: Generate load scenarios
    Prepare->>Prepare: Initialize topology generator
    Prepare->>Prepare: Initialize gen/admittance generators
    Prepare->>Prepare: Network preprocessing
    Prepare-->>CLI: net, load_scenarios, generators

    alt Sequential Mode (Small datasets)
        CLI->>Process: Single-threaded loop
        loop For each scenario
            Process->>Process: Apply perturbations
            Process->>Process: Run OPF/PF solver
            Process->>Process: Validate power bounds
            Process->>Process: Validate power balance
            alt Validation passes
                Process->>Process: Accumulate results
                Note right of Process: Impact: Only valid data included
            else Validation fails
                Process->>Process: Log error, skip scenario
                Note right of Process: Impact: Prevents bad training data
            end
        end
        Process-->>CLI: node_data, edge_data, stats
    else Distributed Mode (1M+ samples)
        CLI->>Process: Multiprocessing with chunks
        Note right of CLI: Impact: Parallel processing<br/>for production scale
        loop For each chunk
            par Worker Processes
                Process->>Process: Apply perturbations
                Process->>Process: Run OPF/PF solver
                Process->>Process: Validate results
            end
            Process->>Save: Intermediate save (chunk)
            Note right of Save: Impact: Memory-efficient<br/>prevents OOM crashes
        end
        Process-->>CLI: Accumulated results
    end

    CLI->>Save: Save all outputs
    Save->>Output: pf_node.csv (bus data)
    Save->>Output: pf_edge.csv (branch data)
    Save->>Output: branch_idx_removed.csv
    Save->>Output: edge_params.csv, bus_params.csv
    Save->>Output: scenarios_*.csv, *.html
    Save->>Output: stats.csv, stats_plot.html
    Save->>Output: logs (tqdm, error, args)
    Note right of Output: Impact: Complete dataset ready<br/>for ML training workflow

    Output-->>User: ✓ Dataset generation complete
    Note left of User: Can now train foundation<br/>models on validated data
```

## Key Insights

- **Mode selection impact**: Sequential for quick iteration (<10K samples), distributed for production (1M+)
- **Memory management**: Chunked saves in distributed mode prevent OOM crashes on large datasets
- **Quality assurance**: Multi-stage validation (convergence, bounds, balance) filters invalid physics
- **Reproducibility**: All logs and parameters saved for experiment tracking
- **Error resilience**: Per-scenario error logging allows partial dataset recovery

## Change History

- **2025-11-10:** Initial feature diagram created
