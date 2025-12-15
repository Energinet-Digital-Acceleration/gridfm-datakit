"""Main data generation module for gridfm_datakit."""

import numpy as np
import os
from gridfm_datakit.save import (
    save_edge_params,
    save_bus_params,
    save_branch_idx_removed,
    save_node_edge_data,
)
from gridfm_datakit.process.process_network import (
    network_preprocessing,
    process_scenario,
    process_scenario_contingency,
    process_scenario_chunk,
    process_scenario_pypowsybl,
    process_scenario_chunk_pypowsybl,
)
from gridfm_datakit.utils.stats import (
    plot_stats,
    Stats,
    plot_feature_distributions,
)
from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_topology_generator,
    initialize_generation_generator,
    initialize_admittance_generator,
)
from gridfm_datakit.network import (
    load_net_from_pp,
    load_net_from_file,
    load_net_from_pglib,
    load_net_from_pypowsybl,
    PyPowSyBlNetwork,
)
from gridfm_datakit.perturbations.load_perturbation import (
    load_scenarios_to_df,
    plot_load_scenarios_combined,
)
from pandapower.auxiliary import pandapowerNet
import gc
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager
import shutil
from gridfm_datakit.utils.utils import write_ram_usage_distributed, Tee
import yaml
from typing import List, Tuple, Any, Dict, Optional, Union
import sys


def _setup_environment(
    config: Union[str, Dict, NestedNamespace],
) -> Tuple[NestedNamespace, str, Dict[str, str]]:
    """Setup the environment for data generation.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)

    Returns:
        Tuple of (args, base_path, file_paths)
    """
    # Load config from file if a path is provided
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    # Convert dict to NestedNamespace if needed
    if isinstance(config, dict):
        args = NestedNamespace(**config)
    else:
        args = config

    # Setup output directory
    base_path = os.path.join(args.settings.data_dir, args.network.name, "raw")
    if os.path.exists(base_path) and args.settings.overwrite:
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)

    # Setup file paths
    file_paths = {
        "tqdm_log": os.path.join(base_path, "tqdm.log"),
        "error_log": os.path.join(base_path, "error.log"),
        "args_log": os.path.join(base_path, "args.log"),
        "node_data": os.path.join(base_path, "pf_node.csv"),
        "edge_data": os.path.join(base_path, "pf_edge.csv"),
        "branch_indices": os.path.join(base_path, "branch_idx_removed.csv"),
        "edge_params": os.path.join(base_path, "edge_params.csv"),
        "bus_params": os.path.join(base_path, "bus_params.csv"),
        "scenarios": os.path.join(base_path, f"scenarios_{args.load.generator}.csv"),
        "scenarios_plot": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.html",
        ),
        "scenarios_log": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.log",
        ),
        "feature_plots": os.path.join(base_path, "feature_plots"),
    }

    # Initialize logs
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for log_file in [
        file_paths["tqdm_log"],
        file_paths["error_log"],
        file_paths["scenarios_log"],
        file_paths["args_log"],
    ]:
        with open(log_file, "a") as f:
            f.write(f"\nNew generation started at {timestamp}\n")
            if log_file == file_paths["args_log"]:
                yaml.dump(config if isinstance(config, dict) else vars(config), f)

    return args, base_path, file_paths


def _prepare_network_and_scenarios(
    args: NestedNamespace,
    file_paths: Dict[str, str],
) -> Tuple[Union[pandapowerNet, PyPowSyBlNetwork], Any]:
    """Prepare the network and generate load scenarios.

    Args:
        args: Configuration object
        file_paths: Dictionary of file paths

    Returns:
        Tuple of (network, scenarios) where network is pandapowerNet or PyPowSyBlNetwork
    """
    # Load network
    if args.network.source == "pandapower":
        net = load_net_from_pp(args.network.name)
    elif args.network.source == "pglib":
        net = load_net_from_pglib(args.network.name)
    elif args.network.source == "pypowsybl":
        # Native pypowsybl path - returns PyPowSyBlNetwork
        psy_net = load_net_from_pypowsybl(args.network.name)
        scenarios = _prepare_pypowsybl_scenarios(psy_net, args, file_paths)
        return psy_net, scenarios
    elif args.network.source == "file":
        net = load_net_from_file(
            os.path.join(args.network.network_dir, args.network.name) + ".m",
        )
    else:
        raise ValueError("Invalid grid source!")

    network_preprocessing(net)
    assert (net.sgen["scaling"] == 1).all(), "Scaling factor >1 not supported yet!"

    # Generate load scenarios
    load_scenario_generator = get_load_scenario_generator(args.load)
    scenarios = load_scenario_generator(
        net,
        args.load.scenarios,
        file_paths["scenarios_log"],
    )
    scenarios_df = load_scenarios_to_df(scenarios)
    scenarios_df.to_csv(file_paths["scenarios"], index=False)
    plot_load_scenarios_combined(scenarios_df, file_paths["scenarios_plot"])
    save_edge_params(net, file_paths["edge_params"])
    save_bus_params(net, file_paths["bus_params"])

    return net, scenarios


def _prepare_pypowsybl_scenarios(
    psy_net: PyPowSyBlNetwork,
    args: NestedNamespace,
    file_paths: Dict[str, str],
) -> np.ndarray:
    """Prepare load scenarios for pypowsybl network.

    Args:
        psy_net: PyPowSyBlNetwork container
        args: Configuration object
        file_paths: Dictionary of file paths

    Returns:
        numpy.ndarray: Load scenarios of shape (n_loads, n_scenarios, 2)
    """
    import pandas as pd

    # Get base loads from pypowsybl network
    loads = psy_net.network.get_loads()
    n_loads = len(loads)
    n_scenarios = args.load.scenarios

    # Get base P and Q values
    base_p = loads["p0"].values
    base_q = loads["q0"].values

    # Generate scenarios using random scaling (simplified PowerGraph-like approach)
    # Scale loads between 0.7 and 1.3 of base values
    np.random.seed(42)  # For reproducibility
    scaling_factors = np.random.uniform(0.7, 1.3, size=(n_loads, n_scenarios))

    scenarios = np.zeros((n_loads, n_scenarios, 2))
    for i in range(n_scenarios):
        scenarios[:, i, 0] = base_p * scaling_factors[:, i]  # P
        scenarios[:, i, 1] = base_q * scaling_factors[:, i]  # Q (same scaling to maintain power factor)

    # Save scenarios to CSV
    scenarios_df = load_scenarios_to_df(scenarios)
    scenarios_df.to_csv(file_paths["scenarios"], index=False)
    plot_load_scenarios_combined(scenarios_df, file_paths["scenarios_plot"])

    # Note: edge_params and bus_params require pandapower format
    # For pypowsybl, we skip these or generate simplified versions
    # TODO: Implement pypowsybl-specific edge/bus params if needed

    return scenarios


def _save_generated_data(
    net: Union[pandapowerNet, PyPowSyBlNetwork],
    csv_data: List,
    adjacency_lists: List,
    branch_idx_removed: List,
    global_stats: Optional[Stats],
    file_paths: Dict[str, str],
    base_path: str,
    args: NestedNamespace,
) -> None:
    """Save the generated data to files.

    Args:
        net: Pandapower network or PyPowSyBlNetwork
        csv_data: List of CSV data
        adjacency_lists: List of adjacency lists
        branch_idx_removed: List of removed branch indices
        global_stats: Optional statistics object
        file_paths: Dictionary of file paths
        base_path: Base output directory
        args: Configuration object
    """
    if len(adjacency_lists) > 0:
        # For pypowsybl, use simplified save (no pandapower-specific data)
        if isinstance(net, PyPowSyBlNetwork):
            _save_pypowsybl_data(
                net,
                file_paths["node_data"],
                file_paths["edge_data"],
                csv_data,
                adjacency_lists,
            )
        else:
            save_node_edge_data(
                net,
                file_paths["node_data"],
                file_paths["edge_data"],
                csv_data,
                adjacency_lists,
                mode=args.settings.mode,
            )
            save_branch_idx_removed(branch_idx_removed, file_paths["branch_indices"])
        if not args.settings.no_stats and global_stats:
            global_stats.save(base_path)
            plot_stats(base_path)


def _save_pypowsybl_data(
    psy_net: PyPowSyBlNetwork,
    node_path: str,
    edge_path: str,
    csv_data: List,
    adjacency_lists: List,
) -> None:
    """Save pypowsybl power flow data to CSV files.

    Args:
        psy_net: PyPowSyBlNetwork container
        node_path: Path to save node data
        edge_path: Path to save edge data
        csv_data: List of node data arrays
        adjacency_lists: List of adjacency list arrays
    """
    import pandas as pd

    # Node data columns (same as pandapower output)
    node_columns = ["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]

    # Reshape csv_data: each scenario has n_buses rows
    n_buses = psy_net.n_buses
    n_scenarios = len(csv_data) // n_buses if csv_data else 0

    if n_scenarios > 0:
        csv_array = np.array(csv_data).reshape(n_scenarios, n_buses, -1)
        rows = []
        for scen_idx in range(n_scenarios):
            for bus_idx in range(n_buses):
                row = [scen_idx] + list(csv_array[scen_idx, bus_idx, :])
                rows.append(row)

        node_df = pd.DataFrame(rows, columns=node_columns)
        node_df.to_csv(node_path, index=False)

    # Edge data
    edge_columns = ["scenario", "index1", "index2", "G", "B"]
    edge_rows = []
    for scen_idx, adj_list in enumerate(adjacency_lists):
        for row in adj_list:
            edge_rows.append([scen_idx] + list(row))

    if edge_rows:
        edge_df = pd.DataFrame(edge_rows, columns=edge_columns)
        edge_df.to_csv(edge_path, index=False)


def generate_power_flow_data(
    config: Union[str, Dict, NestedNamespace],
) -> Dict[str, str]:
    """Generate power flow data based on the provided configuration using sequential processing.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)
            The config must include settings, network, load, and topology_perturbation configurations.

    Returns:
        Dictionary containing paths to generated files:
        {
            'node_data': path to node data CSV,
            'edge_data': path to edge data CSV,
            'branch_indices': path to branch indices CSV,
            'edge_params': path to edge parameters CSV,
            'bus_params': path to bus parameters CSV,
            'scenarios': path to scenarios CSV,
            'scenarios_plot': path to scenarios plot HTML,
            'scenarios_log': path to scenarios log
        }

    Note:
        The function creates several output files in the specified data directory:

        - tqdm.log: Progress tracking
        - error.log: Error messages
        - args.log: Configuration parameters
        - pf_node.csv: Node data
        - pf_edge.csv: Edge data
        - branch_idx_removed.csv: Removed branch indices
        - edge_params.csv: Edge parameters
        - bus_params.csv: Bus parameters
        - scenarios_{generator}.csv: Load scenarios
        - scenarios_{generator}.html: Scenario plots
        - scenarios_{generator}.log: Scenario generation log
    """
    # Setup environment
    args, base_path, file_paths = _setup_environment(config)

    # Prepare network and scenarios
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

    # Check if using pypowsybl native path
    is_pypowsybl = isinstance(net, PyPowSyBlNetwork)

    csv_data = []
    adjacency_lists = []
    branch_idx_removed = []
    global_stats = Stats() if not args.settings.no_stats else None

    if is_pypowsybl:
        # PyPowSyBl native path - simplified processing
        if args.settings.mode != "pf":
            raise ValueError("pypowsybl source only supports mode='pf'")
        if args.topology_perturbation.type != "none":
            raise ValueError("pypowsybl source does not support topology perturbation")

        with open(file_paths["tqdm_log"], "a") as f:
            with tqdm(
                total=args.load.scenarios,
                desc="Processing scenarios (pypowsybl)",
                file=Tee(sys.stdout, f),
                miniters=5,
            ) as pbar:
                for scenario_index in range(args.load.scenarios):
                    csv_data, adjacency_lists = process_scenario_pypowsybl(
                        net,
                        scenarios,
                        scenario_index,
                        csv_data,
                        adjacency_lists,
                        file_paths["error_log"],
                    )
                    pbar.update(1)
    else:
        # Pandapower path - full processing with perturbations
        topology_generator = initialize_topology_generator(args.topology_perturbation, net)
        generation_generator = initialize_generation_generator(
            args.generation_perturbation,
            net,
        )
        admittance_generator = initialize_admittance_generator(
            args.admittance_perturbation,
            net,
        )

        with open(file_paths["tqdm_log"], "a") as f:
            with tqdm(
                total=args.load.scenarios,
                desc="Processing scenarios",
                file=Tee(sys.stdout, f),
                miniters=5,
            ) as pbar:
                for scenario_index in range(args.load.scenarios):
                    if args.settings.mode == "pf":
                        csv_data, adjacency_lists, branch_idx_removed, global_stats = (
                            process_scenario(
                                net,
                                scenarios,
                                scenario_index,
                                topology_generator,
                                generation_generator,
                                admittance_generator,
                                args.settings.no_stats,
                                csv_data,
                                adjacency_lists,
                                branch_idx_removed,
                                global_stats,
                                file_paths["error_log"],
                            )
                        )
                    elif args.settings.mode == "contingency":
                        csv_data, adjacency_lists, branch_idx_removed, global_stats = (
                            process_scenario_contingency(
                                net,
                                scenarios,
                                scenario_index,
                                topology_generator,
                                generation_generator,
                                admittance_generator,
                                args.settings.no_stats,
                                csv_data,
                                adjacency_lists,
                                branch_idx_removed,
                                global_stats,
                                file_paths["error_log"],
                            )
                        )
                    pbar.update(1)

    # Save final data
    _save_generated_data(
        net,
        csv_data,
        adjacency_lists,
        branch_idx_removed,
        global_stats,
        file_paths,
        base_path,
        args,
    )

    # Plot features (skip for pypowsybl - no sn_mva attribute)
    if not is_pypowsybl and os.path.exists(file_paths["node_data"]):
        plot_feature_distributions(
            file_paths["node_data"],
            file_paths["feature_plots"],
            net.sn_mva,
        )
    elif not os.path.exists(file_paths["node_data"]):
        print("No node data file generated. Skipping feature plotting.")

    return file_paths


def _generate_pypowsybl_sequential(
    psy_net: PyPowSyBlNetwork,
    scenarios: np.ndarray,
    args: NestedNamespace,
    file_paths: Dict[str, str],
    base_path: str,
) -> Dict[str, str]:
    """Sequential processing for pypowsybl networks (helper for distributed fallback).

    Args:
        psy_net: PyPowSyBlNetwork container
        scenarios: Load scenarios array
        args: Configuration object
        file_paths: Dictionary of file paths
        base_path: Base output directory

    Returns:
        Dictionary of file paths
    """
    csv_data = []
    adjacency_lists = []

    with open(file_paths["tqdm_log"], "a") as f:
        with tqdm(
            total=args.load.scenarios,
            desc="Processing scenarios (pypowsybl)",
            file=Tee(sys.stdout, f),
            miniters=5,
        ) as pbar:
            for scenario_index in range(args.load.scenarios):
                csv_data, adjacency_lists = process_scenario_pypowsybl(
                    psy_net,
                    scenarios,
                    scenario_index,
                    csv_data,
                    adjacency_lists,
                    file_paths["error_log"],
                )
                pbar.update(1)

    _save_generated_data(
        psy_net,
        csv_data,
        adjacency_lists,
        [],  # branch_idx_removed
        None,  # global_stats
        file_paths,
        base_path,
        args,
    )

    return file_paths


def generate_power_flow_data_distributed(
    config: Union[str, Dict, NestedNamespace],
) -> Dict[str, str]:
    """Generate power flow data based on the provided configuration using distributed processing.


    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)
            The config must include settings, network, load, and topology_perturbation configurations.

    Returns:
        Dictionary containing paths to generated files:
        {
            'node_data': path to node data CSV,
            'edge_data': path to edge data CSV,
            'branch_indices': path to branch indices CSV,
            'edge_params': path to edge parameters CSV,
            'bus_params': path to bus parameters CSV,
            'scenarios': path to scenarios CSV,
            'scenarios_plot': path to scenarios plot HTML,
            'scenarios_log': path to scenarios log
        }

    Note:
        The function creates several output files in the specified data directory:

        - tqdm.log: Progress tracking
        - error.log: Error messages
        - args.log: Configuration parameters
        - pf_node.csv: Node data
        - pf_edge.csv: Edge data
        - branch_idx_removed.csv: Removed branch indices
        - edge_params.csv: Edge parameters
        - bus_params.csv: Bus parameters
        - scenarios_{generator}.csv: Load scenarios
        - scenarios_{generator}.html: Scenario plots
        - scenarios_{generator}.log: Scenario generation log
    """
    # Setup environment
    args, base_path, file_paths = _setup_environment(config)

    # Prepare network and scenarios
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

    # Check if using pypowsybl - fall back to sequential (pypowsybl networks not picklable)
    if isinstance(net, PyPowSyBlNetwork):
        print("Note: pypowsybl source uses sequential processing (multiprocessing not supported)")
        return _generate_pypowsybl_sequential(net, scenarios, args, file_paths, base_path)

    # Initialize topology generator
    topology_generator = initialize_topology_generator(args.topology_perturbation, net)

    # Initialize generation generator
    generation_generator = initialize_generation_generator(
        args.generation_perturbation,
        net,
    )

    # Initialize admittance generator
    admittance_generator = initialize_admittance_generator(
        args.admittance_perturbation,
        net,
    )

    # Setup multiprocessing
    manager = Manager()
    progress_queue = manager.Queue()

    # Process scenarios in chunks
    large_chunks = np.array_split(
        range(args.load.scenarios),
        np.ceil(args.load.scenarios / args.settings.large_chunk_size).astype(int),
    )

    with open(file_paths["tqdm_log"], "a") as f:
        with tqdm(
            total=args.load.scenarios,
            desc="Processing scenarios",
            file=Tee(sys.stdout, f),
            miniters=5,
        ) as pbar:
            for large_chunk_index, large_chunk in enumerate(large_chunks):
                write_ram_usage_distributed(f)
                chunk_size = len(large_chunk)
                scenario_chunks = np.array_split(
                    large_chunk,
                    args.settings.num_processes,
                )

                tasks = [
                    (
                        args.settings.mode,
                        chunk[0],
                        chunk[-1] + 1,
                        scenarios,
                        net,
                        progress_queue,
                        topology_generator,
                        generation_generator,
                        admittance_generator,
                        args.settings.no_stats,
                        file_paths["error_log"],
                    )
                    for chunk in scenario_chunks
                ]

                # Run parallel processing
                with Pool(processes=args.settings.num_processes) as pool:
                    results = [
                        pool.apply_async(process_scenario_chunk, task) for task in tasks
                    ]

                    # Update progress
                    completed = 0
                    while completed < chunk_size:
                        progress_queue.get()
                        pbar.update(1)
                        completed += 1

                    # Gather results
                    csv_data = []
                    adjacency_lists = []
                    branch_idx_removed = []
                    global_stats = Stats() if not args.settings.no_stats else None

                    for result in results:
                        (
                            e,
                            traceback,
                            local_csv_data,
                            local_adjacency_lists,
                            local_branch_idx_removed,
                            local_stats,
                        ) = result.get()
                        if isinstance(e, Exception):
                            print(f"Error in process_scenario_chunk: {e}")
                            print(traceback)
                            sys.exit(1)
                        csv_data.extend(local_csv_data)
                        adjacency_lists.extend(local_adjacency_lists)
                        branch_idx_removed.extend(local_branch_idx_removed)
                        if not args.settings.no_stats and local_stats:
                            global_stats.merge(local_stats)

                    pool.close()
                    pool.join()

                # Save processed data
                _save_generated_data(
                    net,
                    csv_data,
                    adjacency_lists,
                    branch_idx_removed,
                    global_stats,
                    file_paths,
                    base_path,
                    args,
                )

                del csv_data, adjacency_lists, global_stats
                gc.collect()

    # Plot features
    # check if node_data csv file exists
    if os.path.exists(file_paths["node_data"]):
        plot_feature_distributions(
            file_paths["node_data"],
            file_paths["feature_plots"],
            net.sn_mva,
        )
    else:
        print("No node data file generated. Skipping feature plotting.")

    return file_paths
