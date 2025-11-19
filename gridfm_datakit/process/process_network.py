import numpy as np
import pandas as pd
from gridfm_datakit.utils.config import PQ, PV, REF
from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
from typing import Tuple, List, Union
from pandapower import makeYbus_pypower
import pandapower as pp
import copy
from gridfm_datakit.process.solvers import run_opf, run_pf
from gridfm_datakit.process.solver_interface import SolverInterface
from pandapower.pypower.idx_brch import BR_STATUS
from queue import Queue
from gridfm_datakit.utils.stats import Stats
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.perturbations.generator_perturbation import GenerationGenerator
from gridfm_datakit.perturbations.admittance_perturbation import AdmittanceGenerator
from gridfm_datakit.network_interface import NetworkInterface, copy_network
from gridfm_datakit.process.network_utils import check_network_feasibility
import traceback


def network_preprocessing(net: Union[pandapowerNet, Network]) -> None:
    """Adds names to bus dataframe and bus types to load, bus, gen, sgen dataframes.

    This function performs several preprocessing steps:

    1. Assigns names to all network components
    2. Determines bus types (PQ, PV, REF)
    3. Assigns bus types to connected components
    4. Performs validation checks on the network structure

    Args:
        net: The power network to preprocess (pandapower or pypowsybl).

    Raises:
        AssertionError: If network structure violates expected constraints:
            - More than one load per bus
            - REF bus not matching ext_grid connection (pandapower only)
            - PQ bus definition mismatch
    """
    # Create adapter for unified interface
    adapter = NetworkInterface.create_adapter(net)

    # Clean-Up things in Data-Frame // give numbered item names
    buses = adapter.get_buses()
    for i in buses.index:
        buses.at[i, "name"] = "Bus " + str(i)
    adapter.update_buses(buses)

    loads = adapter.get_loads()
    for i in loads.index:
        loads.at[i, "name"] = "Load " + str(i)
    adapter.update_loads(loads)

    sgens = adapter.get_static_generators()
    for i in sgens.index:
        sgens.at[i, "name"] = "Sgen " + str(i)
    adapter.update_static_generators(sgens)

    gens = adapter.get_generators()
    for i in gens.index:
        gens.at[i, "name"] = "Gen " + str(i)
    adapter.update_generators(gens)

    shunts = adapter.get_shunts()
    for i in shunts.index:
        shunts.at[i, "name"] = "Shunt " + str(i)
    adapter.update_shunts(shunts)

    ext_grids = adapter.get_external_grids()
    if len(ext_grids) > 0:
        for i in ext_grids.index:
            ext_grids.at[i, "name"] = "Ext_Grid " + str(i)
        # Only pandapower has ext_grid update
        if isinstance(net, pandapowerNet):
            net.ext_grid = ext_grids

    lines = adapter.get_lines()
    for i in lines.index:
        lines.at[i, "name"] = "Line " + str(i)
    adapter.update_lines(lines)

    trafos = adapter.get_transformers()
    for i in trafos.index:
        trafos.at[i, "name"] = "Trafo " + str(i)
    adapter.update_transformers(trafos)

    # Determine bus types
    num_buses = adapter.get_num_buses()
    buses = adapter.get_buses()
    bus_ids = buses.index.tolist()

    # Create mapping for pypowsybl string IDs to integer indices if needed
    if isinstance(net, Network):
        bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        idx_to_bus_id = {idx: bus_id for idx, bus_id in enumerate(bus_ids)}
    else:
        bus_id_to_idx = {i: i for i in range(num_buses)}
        idx_to_bus_id = {i: i for i in range(num_buses)}

    bus_types = np.zeros(num_buses, dtype=int)

    # Identify slack bus
    slack_bus_ids = adapter.identify_slack_bus()
    if isinstance(net, pandapowerNet):
        assert len(ext_grids) == 1, "Pandapower network must have exactly one ext_grid"
        indices_slack = slack_bus_ids
    else:
        # For pypowsybl, convert slack bus ID to index
        indices_slack = np.array([bus_id_to_idx[bus_id] for bus_id in slack_bus_ids])

    # Get generator and static generator buses
    gens = adapter.get_generators()
    sgens = adapter.get_static_generators()

    if isinstance(net, pandapowerNet):
        # Pandapower uses integer bus indices
        gen_buses = np.unique(np.array(gens["bus"])) if len(gens) > 0 else np.array([], dtype=int)
        sgen_buses = np.unique(np.array(sgens["bus"])) if len(sgens) > 0 else np.array([], dtype=int)
    else:
        # Pypowsybl uses string bus IDs - convert to indices
        gen_buses = np.unique(np.array([bus_id_to_idx[bus_id] for bus_id in gens["bus_id"]], dtype=int)) if len(gens) > 0 else np.array([], dtype=int)
        sgen_buses = np.unique(np.array([bus_id_to_idx[bus_id] for bus_id in sgens["bus_id"]], dtype=int)) if len(sgens) > 0 else np.array([], dtype=int)

    indices_PV = np.union1d(gen_buses, sgen_buses)
    indices_PV = np.setdiff1d(indices_PV, indices_slack)  # Exclude slack

    indices_PQ = np.setdiff1d(
        np.arange(num_buses),
        np.union1d(indices_PV, indices_slack),
    )

    bus_types[indices_PQ] = PQ  # Set PQ bus types to 1
    bus_types[indices_PV] = PV  # Set PV bus types to 2
    bus_types[indices_slack] = REF  # Set Slack bus types to 3

    # Update bus types in network
    buses["type"] = bus_types
    adapter.update_buses(buses)

    # Fetch buses again to get updated values
    buses = adapter.get_buses()

    # Assign type of the bus connected to each load and generator
    loads = adapter.get_loads()
    gens = adapter.get_generators()
    sgens = adapter.get_static_generators()

    if isinstance(net, pandapowerNet):
        loads["type"] = buses.type[loads.bus].to_list()
        gens["type"] = buses.type[gens.bus].to_list()
        sgens["type"] = buses.type[sgens.bus].to_list()
    else:
        # For pypowsybl, map bus_id to type
        bus_id_to_type = buses["type"].to_dict()
        loads["type"] = [bus_id_to_type.get(bus_id, PQ) for bus_id in loads["bus_id"]]
        gens["type"] = [bus_id_to_type.get(bus_id, PQ) for bus_id in gens["bus_id"]]
        sgens["type"] = [bus_id_to_type.get(bus_id, PQ) for bus_id in sgens["bus_id"]]

    adapter.update_loads(loads)
    adapter.update_generators(gens)
    adapter.update_static_generators(sgens)

    # Validation checks
    loads = adapter.get_loads()
    if isinstance(net, pandapowerNet):
        # There is no more than one load per bus
        assert loads.bus.unique().shape[0] == loads.bus.shape[0], "Multiple loads per bus detected"

        # REF bus is bus with ext grid
        ext_grids = adapter.get_external_grids()
        buses = adapter.get_buses()
        assert (
            np.where(buses["type"] == REF)[0] == ext_grids.bus.values
        ).all(), "REF bus doesn't match ext_grid connection"

        # PQ buses are buses with no gen nor ext_grid
        gens = adapter.get_generators()
        sgens = adapter.get_static_generators()
        assert (
            (buses["type"] == PQ)
            == ~np.isin(
                range(buses.shape[0]),
                np.concatenate([ext_grids.bus.values, gens.bus.values, sgens.bus.values]),
            )
        ).all(), "PQ bus definition mismatch"
    else:
        # For pypowsybl, different validation
        # Check unique loads per bus
        load_bus_ids = loads["bus_id"].tolist()
        assert len(load_bus_ids) == len(set(load_bus_ids)), "Multiple loads per bus detected"


def pf_preprocessing(net: Union[pandapowerNet, Network]) -> Union[pandapowerNet, Network]:
    """Sets variables to the results of OPF.

    Updates the following network components with OPF results:

    - sgen.p_mw/target_p: active power generation for static generators
    - gen.p_mw/target_p, gen.vm_pu/target_v: active power and voltage magnitude for generators

    Args:
        net: The power network to preprocess (pandapower or pypowsybl).

    Returns:
        The updated power network with OPF results.
    """
    adapter = NetworkInterface.create_adapter(net)

    if isinstance(net, pandapowerNet):
        # Pandapower: copy from res_* tables to main tables
        net.sgen[["p_mw"]] = net.res_sgen[["p_mw"]]
        net.gen[["p_mw", "vm_pu"]] = net.res_gen[["p_mw", "vm_pu"]]
    else:
        # Pypowsybl: results are in same table, just need to copy p→target_p, etc.
        sgens = adapter.get_static_generators()
        if len(sgens) > 0:
            sgens_updated = sgens.copy()
            if 'p' in sgens_updated.columns:
                sgens_updated['target_p'] = sgens_updated['p']
            adapter.update_static_generators(sgens_updated)

        gens = adapter.get_generators()
        if len(gens) > 0:
            gens_updated = gens.copy()
            if 'p' in gens_updated.columns:
                gens_updated['target_p'] = gens_updated['p']
            if 'v_mag' in gens_updated.columns:
                gens_updated['target_v'] = gens_updated['v_mag']
            adapter.update_generators(gens_updated)

    return net


def pf_post_processing(net: Union[pandapowerNet, Network], dcpf: bool = False) -> np.ndarray:
    """Post-processes PF data to build the final data representation.

    Creates a matrix of shape (n_buses, 10) or (n_buses, 12) for DC power flow,
    with columns: (bus, Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF) plus (Vm_dc, Va_dc)
    for DC power flow.

    Args:
        net: The power network to process (pandapower or pypowsybl).
        dcpf: Whether to include DC power flow results. Defaults to False.

    Returns:
        numpy.ndarray: Matrix containing the processed power flow data.
    """
    adapter = NetworkInterface.create_adapter(net)

    buses = adapter.get_buses()
    num_buses = len(buses)
    X = np.zeros((num_buses, 12 if dcpf else 10))

    if isinstance(net, pandapowerNet):
        # Pandapower path (original logic)
        all_loads = (
            pd.concat([net.res_load])[["p_mw", "q_mvar", "bus"]].groupby("bus").sum()
        )

        all_gens = (
            pd.concat([net.res_gen, net.res_sgen, net.res_ext_grid])[
                ["p_mw", "q_mvar", "bus"]
            ]
            .groupby("bus")
            .sum()
        )

        assert (net.bus.index.values == list(range(X.shape[0]))).all()

        X[:, 0] = net.bus.index.values

        # Active and reactive power demand
        X[all_loads.index, 1] = all_loads.p_mw  # Pd
        X[all_loads.index, 2] = all_loads.q_mvar  # Qd

        # Active and reactive power generated
        X[net.bus.type == PV, 3] = all_gens.p_mw[
            net.res_bus.type == PV
        ]  # active Power generated
        X[net.bus.type == PV, 4] = all_gens.q_mvar[
            net.res_bus.type == PV
        ]  # reactive Power generated
        X[net.bus.type == REF, 3] = all_gens.p_mw[
            net.res_bus.type == REF
        ]  # active Power generated
        X[net.bus.type == REF, 4] = all_gens.q_mvar[
            net.res_bus.type == REF
        ]  # reactive Power generated

        # Voltage
        X[:, 5] = net.res_bus.vm_pu  # voltage magnitude
        X[:, 6] = net.res_bus.va_degree  # voltage angle
        X[:, 7:10] = pd.get_dummies(net.bus["type"]).values

        if dcpf:
            X[:, 10] = net.bus["Vm_dc"]
            X[:, 11] = net.bus["Va_dc"]
    else:
        # Pypowsybl path
        bus_ids = buses.index.tolist()
        bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

        # Bus indices (using integer index)
        X[:, 0] = np.arange(num_buses)

        # Get loads and aggregate by bus
        loads = adapter.get_result_loads()
        load_power = {}
        for idx, load in loads.iterrows():
            bus_id = load['bus_id']
            bus_idx = bus_id_to_idx[bus_id]
            p = load.get('p', load.get('p0', 0))
            q = load.get('q', load.get('q0', 0))
            if bus_idx in load_power:
                load_power[bus_idx][0] += p
                load_power[bus_idx][1] += q
            else:
                load_power[bus_idx] = [p, q]

        for bus_idx, (p, q) in load_power.items():
            X[bus_idx, 1] = p  # Pd
            X[bus_idx, 2] = q  # Qd

        # Get generators and aggregate by bus
        gens = adapter.get_result_generators()
        sgens = adapter.get_result_static_generators()
        ext_grids = adapter.get_result_external_grids()

        all_gens = pd.concat([gens, sgens, ext_grids]) if len(ext_grids) > 0 else pd.concat([gens, sgens])

        gen_power = {}
        for idx, gen in all_gens.iterrows():
            bus_id = gen.get('bus_id', gen.get('bus'))
            if bus_id not in bus_id_to_idx:
                continue
            bus_idx = bus_id_to_idx[bus_id]
            p = gen.get('p', gen.get('target_p', 0))
            q = gen.get('q', gen.get('target_q', 0))
            if bus_idx in gen_power:
                gen_power[bus_idx][0] += p
                gen_power[bus_idx][1] += q
            else:
                gen_power[bus_idx] = [p, q]

        # Assign generation based on bus type
        bus_types = buses['type'].values
        for bus_idx, (p, q) in gen_power.items():
            if bus_types[bus_idx] == PV or bus_types[bus_idx] == REF:
                X[bus_idx, 3] = p  # Pg
                X[bus_idx, 4] = q  # Qg

        # Voltage magnitude and angle
        result_buses = adapter.get_result_buses()
        X[:, 5] = result_buses['v_mag'].values if 'v_mag' in result_buses else 1.0
        X[:, 6] = result_buses['v_angle'].values if 'v_angle' in result_buses else 0.0

        # Bus types as one-hot encoding
        X[:, 7:10] = pd.get_dummies(buses["type"]).values

        if dcpf:
            X[:, 10] = buses.get("Vm_dc", 1.0) if "Vm_dc" in buses else 1.0
            X[:, 11] = buses.get("Va_dc", 0.0) if "Va_dc" in buses else 0.0

    return X


def get_adjacency_list(net: Union[pandapowerNet, Network]) -> np.ndarray:
    """Gets adjacency list for network.

    Creates an adjacency list representation of the network's bus admittance matrix,
    including real and imaginary components of the admittance.

    Args:
        net: The power network (pandapower or pypowsybl).

    Returns:
        numpy.ndarray: Array containing edge indices and attributes (G, B).
    """
    adapter = NetworkInterface.create_adapter(net)
    Y_bus, Yf, Yt = adapter.get_ybus()

    i, j = np.nonzero(Y_bus)
    # note that Y_bus[i,j] can be != 0 even if a branch from i to j is not in service because there might be other branches connected to the same buses

    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    adjacency_lists = np.column_stack((edge_index, edge_attr))
    return adjacency_lists


def get_branch_idx_removed(net: Union[pandapowerNet, Network]) -> List[int]:
    """Gets indices of removed branches in the network.

    Args:
        net: The power network (pandapower or pypowsybl).

    Returns:
        List of indices of branches that are out of service (= removed when applying topology perturbations)
    """
    adapter = NetworkInterface.create_adapter(net)
    ppc = adapter.get_ppc()
    branch = ppc["branch"]
    in_service = branch[:, BR_STATUS]
    return np.where(in_service == 0)[0].tolist()


def process_scenario_contingency(
    net: Union[pandapowerNet, Network],
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    no_stats: bool,
    local_csv_data: List[np.ndarray],
    local_adjacency_lists: List[np.ndarray],
    local_branch_idx_removed: List[List[int]],
    local_stats: Union[Stats, None],
    error_log_file: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], Union[Stats, None]]:
    """Processes a load scenario for contingency analysis.

    Args:
        net: The power network.
        scenarios: Array of load scenarios.
        scenario_index: Index of the current scenario.
        topology_generator: Topology perturbation generator.
        generation_generator: Generator cost perturbation generator.
        admittance_generator: Line admittance perturbation generator.
        no_stats: Whether to skip statistics collection.
        local_csv_data: List to store processed CSV data.
        local_adjacency_lists: List to store adjacency lists.
        local_branch_idx_removed: List to store removed branch indices.
        local_stats: Statistics object for collecting network statistics.
        error_log_file: Path to error log file.

    Returns:
        Tuple containing:
            - List of processed CSV data
            - List of adjacency lists
            - List of removed branch indices
            - Statistics object
    """
    net = copy_network(net)

    # apply the load scenario to the network
    adapter = NetworkInterface.create_adapter(net)
    loads = adapter.get_loads()
    if isinstance(net, pandapowerNet):
        loads['p_mw'] = scenarios[:, scenario_index, 0]
        loads['q_mvar'] = scenarios[:, scenario_index, 1]
    else:  # pypowsybl
        loads['p0'] = scenarios[:, scenario_index, 0]
        loads['q0'] = scenarios[:, scenario_index, 1]
    adapter.update_loads(loads)

    # For pandapower: run OPF to get the gen set points
    # For pypowsybl: skip OPF (not supported), use existing setpoints
    if isinstance(net, pandapowerNet):
        solver = SolverInterface.create_solver(net)
        try:
            solver.run_opf()
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
                )
            return (
                local_csv_data,
                local_adjacency_lists,
                local_branch_idx_removed,
                local_stats,
            )

    net_pf = copy_network(net)
    net_pf = pf_preprocessing(net_pf)

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net_pf)

    # Apply generation perturbations
    perturbations = generation_generator.generate(perturbations)

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    # to simulate contingency, we apply the topology perturbation after OPF
    for perturbation in perturbations:
        perturbation_solver = SolverInterface.create_solver(perturbation)
        try:
            # run DCPF for benchmarking purposes
            perturbation_solver.run_dcpf()
            # Store DC results
            pert_adapter = NetworkInterface.create_adapter(perturbation)
            buses = pert_adapter.get_buses()
            buses["Vm_dc"] = buses.get('v_mag', buses.get('vm_pu', 1.0))
            buses["Va_dc"] = buses.get('v_angle', buses.get('va_degree', 0.0))
            pert_adapter.update_buses(buses)
            # run AC-PF to get the new state of the network after contingency (we don't model any remedial actions)
            perturbation_solver.run_pf()
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} when solving dcpf or in in run_pf function: {e}\n",
                )

                continue

                # TODO: What to do when the network does not converge for AC-PF? -> we dont have targets for regression!!

        # Append processed power flow data
        local_csv_data.extend(pf_post_processing(perturbation, dcpf=True))
        local_adjacency_lists.append(get_adjacency_list(perturbation))
        local_branch_idx_removed.append(get_branch_idx_removed(perturbation))
        if not no_stats:
            local_stats.update(perturbation)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats


def process_scenario_chunk(
    mode,
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    net: Union[pandapowerNet, Network],
    progress_queue: Queue,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    no_stats: bool,
    error_log_path,
) -> Tuple[
    Union[None, Exception],
    Union[None, str],
    List[np.ndarray],
    List[np.ndarray],
    List[List[int]],
    Union[Stats, None],
]:
    """
    Create data for all scenarios in scenario indexed between start_idx and end_idx
    """
    try:
        # For pypowsybl networks, ensure custom attributes are initialized
        # (they may be lost during pickling for multiprocessing)
        if isinstance(net, Network):
            if not hasattr(net, '_gridfm_custom_attrs'):
                net._gridfm_custom_attrs = {
                    'bus': {}, 'load': {}, 'gen': {}, 'line': {}, 'trafo': {}, 'shunt': {}
                }
            # Re-run preprocessing to restore bus types and other metadata
            # This is needed because custom attributes don't survive pickling
            network_preprocessing(net)

        local_stats = Stats() if not no_stats else None
        local_csv_data = []
        local_adjacency_lists = []
        local_branch_idx_removed = []
        for scenario_index in range(start_idx, end_idx):
            if mode == "pf":
                (
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                ) = process_scenario(
                    net,
                    scenarios,
                    scenario_index,
                    topology_generator,
                    generation_generator,
                    admittance_generator,
                    no_stats,
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                    error_log_path,
                )
            elif mode == "contingency":
                (
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                ) = process_scenario_contingency(
                    net,
                    scenarios,
                    scenario_index,
                    topology_generator,
                    generation_generator,
                    admittance_generator,
                    no_stats,
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                    error_log_path,
                )

            progress_queue.put(1)  # update queue

        return (
            None,
            None,
            local_csv_data,
            local_adjacency_lists,
            local_branch_idx_removed,
            local_stats,
        )
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Caught an exception in process_scenario_chunk function: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        for _ in range(end_idx - start_idx):
            progress_queue.put(1)
        return e, traceback.format_exc(), None, None, None, None


def process_scenario(
    net: Union[pandapowerNet, Network],
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    no_stats: bool,
    local_csv_data: List[np.ndarray],
    local_adjacency_lists: List[np.ndarray],
    local_branch_idx_removed: List[List[int]],
    local_stats: Union[Stats, None],
    error_log_file: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], Union[Stats, None]]:
    """Processes a load scenario.

    Args:
        net: The power network.
        scenarios: Array of load scenarios.
        scenario_index: Index of the current scenario.
        topology_generator: Topology perturbation generator.
        generation_generator: Generator cost perturbation generator.
        admittance_generator: Line admittance perturbation generator.
        no_stats: Whether to skip statistics collection.
        local_csv_data: List to store processed CSV data.
        local_adjacency_lists: List to store adjacency lists.
        local_branch_idx_removed: List to store removed branch indices.
        local_stats: Statistics object for collecting network statistics.
        error_log_file: Path to error log file.

    Returns:
        Tuple containing:
            - List of processed CSV data
            - List of adjacency lists
            - List of removed branch indices
            - Statistics object
    """
    # apply the load scenario to the network
    adapter = NetworkInterface.create_adapter(net)
    loads = adapter.get_loads()
    if isinstance(net, pandapowerNet):
        loads['p_mw'] = scenarios[:, scenario_index, 0]
        loads['q_mvar'] = scenarios[:, scenario_index, 1]
    else:  # pypowsybl
        loads['p0'] = scenarios[:, scenario_index, 0]
        loads['q0'] = scenarios[:, scenario_index, 1]
    adapter.update_loads(loads)

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net)

    # Apply generation perturbations
    perturbations = generation_generator.generate(perturbations)

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    for perturbation in perturbations:
        # For pandapower: run OPF to optimize generation setpoints
        # For pypowsybl: skip OPF (not supported), use existing setpoints
        if isinstance(net, pandapowerNet):
            perturbation_solver = SolverInterface.create_solver(perturbation)
            try:
                # run OPF to get the gen set points. Here the set points account for the topology perturbation.
                perturbation_solver.run_opf()
            except Exception as e:
                with open(error_log_file, "a") as f:
                    f.write(
                        f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
                    )
                continue

        net_pf = copy_network(perturbation)

        net_pf = pf_preprocessing(net_pf)

        net_pf_solver = SolverInterface.create_solver(net_pf)
        try:
            # This is not striclty necessary as we havent't changed the setpoints nor the load since we solved OPF, but it gives more accurate PF results
            net_pf_solver.run_pf()

        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_pf function: {e}\n",
                )
            continue

        # Validation checks (pandapower-specific)
        if isinstance(net, pandapowerNet):
            assert (
                net.res_line.loc[net.line.in_service == 0, "loading_percent"] == 0
            ).all(), "Line loading percent is not 0 where the line is not in service"
            assert (
                net.res_trafo.loc[net.trafo.in_service == 0, "loading_percent"] == 0
            ).all(), "Trafo loading percent is not 0 where the trafo is not in service"

            assert (
                net_pf.res_gen.vm_pu[net_pf.res_gen.type == 2]
                - perturbation.res_gen.vm_pu[perturbation.res_gen.type == 2]
                < 1e-3
            ).all(), "Generator voltage at PV buses is not the same after PF"

        # Append processed power flow data
        local_csv_data.extend(pf_post_processing(net_pf))
        local_adjacency_lists.append(get_adjacency_list(net_pf))
        local_branch_idx_removed.append(get_branch_idx_removed(net_pf))
        if not no_stats:
            local_stats.update(net_pf)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats
