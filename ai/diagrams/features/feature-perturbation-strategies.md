# Perturbation Strategies Feature

**Type:** Feature Diagram
**Last Updated:** 2025-11-10
**Related Files:**
- `gridfm_datakit/perturbations/load_perturbation.py`
- `gridfm_datakit/perturbations/topology_perturbation.py`
- `gridfm_datakit/perturbations/generator_perturbation.py`
- `gridfm_datakit/perturbations/admittance_perturbation.py`

## Purpose

Shows researchers how different perturbation strategies create dataset diversity for training robust foundation models that generalize across operating conditions.

## Diagram

```mermaid
flowchart TB
    subgraph "Front-Stage: Training Data Diversity"
        User["👤 ML Researcher<br/>Goal: Train robust models<br/>that generalize"]
        Config["Config Selects:<br/>- Load strategy<br/>- Topology strategy<br/>- Gen/admittance options"]
        Impact["Training Impact:<br/>✓ Learn under uncertainty<br/>✓ Handle contingencies<br/>✓ Robust to variations"]
    end

    subgraph "Back-Stage: Perturbation Engine"
        direction TB

        subgraph "1. Load Perturbations"
            LoadBase["LoadScenarioGeneratorBase<br/>(ABC Pattern)"]
            Aggregate["AggregateLoadProfile:<br/>Historical + noise<br/>Impact: Realistic demand<br/>patterns"]
            PowerGraph["PowerGraph:<br/>Synthetic generation<br/>Impact: Novel scenarios<br/>beyond historical"]
            LoadBase --> Aggregate
            LoadBase --> PowerGraph
        end

        subgraph "2. Topology Perturbations"
            TopoBase["TopologyGenerator<br/>(ABC Pattern)"]
            NoPert["NoPerturbation:<br/>Base case only"]
            NMinusK["N-k Contingency:<br/>All k-element failures<br/>Impact: Safety-critical<br/>what-if analysis"]
            Random["RandomTopology:<br/>Stochastic removal<br/>Impact: Diverse<br/>configurations"]
            TopoBase --> NoPert
            TopoBase --> NMinusK
            TopoBase --> Random

            Feasibility["Feasibility Check:<br/>No unsupplied buses<br/>Impact: Only valid<br/>topologies used"]
            NMinusK --> Feasibility
            Random --> Feasibility
        end

        subgraph "3. Generation Cost Perturbations"
            GenBase["GenerationGenerator<br/>(ABC Pattern)"]
            GenPert["Cost variations<br/>Impact: Economic<br/>dispatch diversity"]
            GenBase --> GenPert
        end

        subgraph "4. Network Parameter Perturbations"
            AdmBase["AdmittanceGenerator<br/>(ABC Pattern)"]
            AdmPert["Line impedance variations<br/>Impact: Physical<br/>uncertainty modeling"]
            AdmBase --> AdmPert
        end

        Combined["Combined Scenarios<br/>Cartesian product of<br/>perturbations<br/>Impact: Rich dataset covering<br/>operating space"]

        Aggregate --> Combined
        PowerGraph --> Combined
        Feasibility --> Combined
        GenPert --> Combined
        AdmPert --> Combined
    end

    User -->|"1. Specify strategies"| Config
    Config -->|"2. Initialize generators"| LoadBase
    Config -->|"2. Initialize generators"| TopoBase
    Config -->|"2. Initialize generators"| GenBase
    Config -->|"2. Initialize generators"| AdmBase

    Combined -->|"3. Generate dataset"| Impact
    Impact -->|"4. Train on diverse data"| User

    style User fill:#e1f5ff
    style Impact fill:#90EE90
    style Feasibility fill:#fff4e1
    style Combined fill:#fff4e1
```

## Key Insights

- **ABC extensibility**: Researchers can implement custom generators without modifying core code
- **N-k contingency value**: Enables safety analysis for grid operators (what if k components fail?)
- **Feasibility filtering**: Prevents training on physically impossible states (buses must have supply)
- **Combinatorial explosion**: N-k with k>1 generates many topologies (use cautiously for large grids)
- **Dataset richness**: Combining strategies creates comprehensive coverage of operating conditions

## Change History

- **2025-11-10:** Initial perturbation strategies diagram created
