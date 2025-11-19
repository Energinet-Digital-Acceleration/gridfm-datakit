# CLI to Dataset User Journey

**Type:** Sequence Diagram
**Last Updated:** 2025-11-10
**Related Files:**
- `gridfm_datakit/cli.py`
- `gridfm_datakit/generate.py`
- `gridfm_datakit/save.py`

## Purpose

Shows the complete user experience from running CLI command to having validated training data, including error handling and progress feedback.

## Diagram

```mermaid
sequenceDiagram
    actor User as 👤 Researcher
    participant CLI as gridfm_datakit CLI
    participant Gen as generate.py
    participant Net as network.py
    participant Pert as Perturbations
    participant Solver as solvers.py
    participant Save as save.py
    participant FS as File System

    Note over User,FS: Impact: One command → production-ready ML dataset

    User->>CLI: $ gridfm_datakit config.yaml
    Note right of User: Config specifies:<br/>- Grid source/name<br/>- Load scenarios<br/>- Topology strategy<br/>- Processing mode

    CLI->>CLI: Validate config YAML
    alt Config invalid
        CLI-->>User: ❌ Config error with details
        Note right of User: Impact: Fast failure<br/>saves compute time
    end

    CLI->>Gen: _setup_environment(config)
    Gen->>FS: Create output dirs
    Gen->>FS: Initialize logs (tqdm, error, args)
    Gen-->>CLI: Paths configured
    Note right of Gen: Impact: Organized outputs<br/>for reproducibility

    CLI->>Gen: _prepare_network_and_scenarios()
    Gen->>Net: load_net_from_*()
    Note right of Net: Supports:<br/>- pandapower<br/>- PGLib (auto-download)<br/>- PyPowSyBl<br/>- MATPOWER files

    Net->>Net: Reindex buses (0...n-1)
    Note right of Net: Critical: Ensures<br/>consistent array ops
    Net-->>Gen: Reindexed network

    Gen->>Pert: Generate load scenarios
    Pert-->>Gen: 3D array (n_loads, n_scenarios, 2)
    Note right of Pert: Shape: (loads, scenarios, [P,Q])

    Gen->>Pert: Initialize topology generator
    Gen->>Pert: Initialize gen/admittance generators
    Gen-->>CLI: Ready to process

    alt Sequential Mode
        Note over CLI,Solver: Small datasets (<10K samples)

        loop For each scenario
            CLI->>Pert: Apply perturbations
            CLI->>Solver: run_opf() or run_pf()

            Solver->>Solver: Solve power flow
            Solver->>Solver: Validate power bounds
            Solver->>Solver: Validate power balance

            alt Solver converges & validates
                Solver-->>CLI: ✓ Valid results
                CLI->>CLI: Accumulate data
            else Solver fails
                Solver-->>CLI: ❌ Convergence/validation failed
                CLI->>FS: Log error with scenario details
                Note right of FS: Impact: Debug failed cases<br/>without losing progress
            end

            CLI->>User: Progress bar update
            Note right of User: Real-time feedback on<br/>generation progress
        end

    else Distributed Mode
        Note over CLI,Solver: Large datasets (1M+ samples)

        CLI->>CLI: Spawn worker pool
        Note right of CLI: Impact: Parallel processing<br/>for production scale

        loop For each chunk
            par Worker 1...N
                CLI->>Pert: Apply perturbations
                CLI->>Solver: run_opf() or run_pf()
                Solver->>Solver: Validate
                alt Valid
                    Solver-->>CLI: Results
                else Invalid
                    Solver-->>CLI: Error (logged)
                end
            end

            CLI->>Save: Intermediate chunk save
            Save->>FS: Append to CSVs
            Note right of FS: Impact: Memory-efficient<br/>prevents OOM crashes

            CLI->>User: Chunk progress update
        end
    end

    CLI->>Gen: _save_generated_data()
    Gen->>Save: Save node/edge data
    Save->>FS: pf_node.csv, pf_edge.csv

    Gen->>Save: Save parameters
    Save->>FS: edge_params.csv, bus_params.csv

    Gen->>Save: Save metadata
    Save->>FS: branch_idx_removed.csv

    Gen->>Save: Save scenarios & plots
    Save->>FS: scenarios_*.csv, *.html

    Gen->>Save: Generate statistics
    Save->>FS: stats.csv, stats_plot.html
    Note right of FS: Impact: Dataset quality<br/>metrics for validation

    Gen-->>CLI: ✓ Generation complete
    CLI-->>User: ✓ Dataset saved to {data_dir}/{network}/raw/

    Note over User: Can now:<br/>✓ Load data for training<br/>✓ Review stats/plots<br/>✓ Check error logs<br/>✓ Reproduce with args.log
```

## Key Insights

- **Fast validation**: Config errors caught before expensive computation starts
- **Progress visibility**: Real-time progress bars reduce user anxiety during long runs
- **Error resilience**: Per-scenario error logging allows partial dataset recovery
- **Auto-download**: PGLib networks fetched automatically (user doesn't need to manage files)
- **Reproducibility**: args.log captures exact configuration for experiment tracking
- **Memory safety**: Chunked distributed mode prevents OOM on million+ sample datasets
- **Quality metrics**: Auto-generated stats help researchers validate dataset before training

## Change History

- **2025-11-10:** Initial user journey diagram created
