import numpy as np
import pandas as pd
from gridfm_datakit.utils.config import PQ, PV, REF
from pandapower.auxiliary import pandapowerNet
from typing import Tuple, List, Union, Dict
from pandapower import makeYbus_pypower
import pandapower as pp
import copy
from gridfm_datakit.process.solvers import run_opf, run_pf, run_pf_pypowsybl
from pandapower.pypower.idx_brch import BR_STATUS
from queue import Queue
from gridfm_datakit.utils.stats import Stats
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.perturbations.generator_perturbation import GenerationGenerator
from gridfm_datakit.perturbations.admittance_perturbation import AdmittanceGenerator
from gridfm_datakit.network import PyPowSyBlNetwork
import traceback


def network_preprocessing(net: pandapowerNet) -> None:
    """Adds names to bus dataframe and bus types to load, bus, gen, sgen dataframes.

    This function performs several preprocessing steps:

    1. Assigns names to all network components
    2. Determines bus types (PQ, PV, REF)
    3. Assigns bus types to connected components
    4. Performs validation checks on the network structure

    Args:
        net: The power network to preprocess.

    Raises:
        AssertionError: If network structure violates expected constraints:
            - More than one load per bus
            - REF bus not matching ext_grid connection
            - PQ bus definition mismatch
    """
    # Clean-Up things in Data-Frame // give numbered item names
    for i, row in net.bus.iterrows():
        net.bus.at[i, "name"] = "Bus " + str(i)
    for i, row in net.load.iterrows():
        net.load.at[i, "name"] = "Load " + str(i)
    for i, row in net.sgen.iterrows():
        net.sgen.at[i, "name"] = "Sgen " + str(i)
    for i, row in net.gen.iterrows():
        net.gen.at[i, "name"] = "Gen " + str(i)
    for i, row in net.shunt.iterrows():
        net.shunt.at[i, "name"] = "Shunt " + str(i)
    for i, row in net.ext_grid.iterrows():
        net.ext_grid.at[i, "name"] = "Ext_Grid " + str(i)
    for i, row in net.line.iterrows():
        net.line.at[i, "name"] = "Line " + str(i)
    for i, row in net.trafo.iterrows():
        net.trafo.at[i, "name"] = "Trafo " + str(i)

    num_buses = len(net.bus)
    bus_types = np.zeros(num_buses, dtype=int)

    # assert one slack bus
    assert len(net.ext_grid) == 1
    indices_slack = np.unique(np.array(net.ext_grid["bus"]))

    indices_PV = np.union1d(
        np.unique(np.array(net.sgen["bus"])),
        np.unique(np.array(net.gen["bus"])),
    )
    indices_PV = np.setdiff1d(
        indices_PV,
        indices_slack,
    )  # Exclude slack indices from PV indices

    indices_PQ = np.setdiff1d(
        np.arange(num_buses),
        np.union1d(indices_PV, indices_slack),
    )

    bus_types[indices_PQ] = PQ  # Set PV bus types to 1
    bus_types[indices_PV] = PV  # Set PV bus types to 2
    bus_types[indices_slack] = REF  # Set Slack bus types to 3

    net.bus["type"] = bus_types

    # assign type of the bus connected to each load and generator
    net.load["type"] = net.bus.type[net.load.bus].to_list()
    net.gen["type"] = net.bus.type[net.gen.bus].to_list()
    net.sgen["type"] = net.bus.type[net.sgen.bus].to_list()

    # there is no more than one load per bus:
    assert net.load.bus.unique().shape[0] == net.load.bus.shape[0]

    # REF bus is bus with ext grid:
    assert (
        np.where(net.bus["type"] == REF)[0]  # REF bus indicated by case file
        == net.ext_grid.bus.values
    ).all()  # Buses connected to an ext grid

    # PQ buses are buses with no gen nor ext_grid, only load or nothing connected to them
    assert (
        (net.bus["type"] == PQ)  # PQ buses indicated by case file
        == ~np.isin(
            range(net.bus.shape[0]),
            np.concatenate(
                [net.ext_grid.bus.values, net.gen.bus.values, net.sgen.bus.values],
            ),
        )
    ).all()  # Buses which are NOT connected to a gen nor an ext grid


def pf_preprocessing(net: pandapowerNet) -> pandapowerNet:
    """Sets variables to the results of OPF.

    Updates the following network components with OPF results:

    - sgen.p_mw: active power generation for static generators
    - gen.p_mw, gen.vm_pu: active power and voltage magnitude for generators

    Args:
        net: The power network to preprocess.

    Returns:
        The updated power network with OPF results.
    """
    net.sgen[["p_mw"]] = net.res_sgen[
        ["p_mw"]
    ]  # sgens are not voltage controlled, so we set P only
    net.gen[["p_mw", "vm_pu"]] = net.res_gen[["p_mw", "vm_pu"]]
    return net


def pf_post_processing(net: pandapowerNet, dcpf: bool = False) -> np.ndarray:
    """Post-processes PF data to build the final data representation.

    Creates a matrix of shape (n_buses, 10) or (n_buses, 12) for DC power flow,
    with columns: (bus, Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF) plus (Vm_dc, Va_dc)
    for DC power flow.

    Args:
        net: The power network to process.
        dcpf: Whether to include DC power flow results. Defaults to False.

    Returns:
        numpy.ndarray: Matrix containing the processed power flow data.
    """
    X = np.zeros((net.bus.shape[0], 12 if dcpf else 10))
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
    return X


def get_adjacency_list(net: pandapowerNet) -> np.ndarray:
    """Gets adjacency list for network.

    Creates an adjacency list representation of the network's bus admittance matrix,
    including real and imaginary components of the admittance.

    Args:
        net: The power network.

    Returns:
        numpy.ndarray: Array containing edge indices and attributes (G, B).
    """
    ppc = net._ppc
    Y_bus, Yf, Yt = makeYbus_pypower(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    i, j = np.nonzero(Y_bus)
    # note that Y_bus[i,j] can be != 0 even if a branch from i to j is not in service because there might be other branches connected to the same buses

    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    adjacency_lists = np.column_stack((edge_index, edge_attr))
    return adjacency_lists


def get_branch_idx_removed(branch: np.ndarray) -> List[int]:
    """Gets indices of removed branches in the network.

    Args:
        branch: Branch data array from the network.

    Returns:
        List of indices of branches that are out of service (= removed when applying topology perturbations)
    """
    in_service = branch[:, BR_STATUS]
    return np.where(in_service == 0)[0].tolist()


# ============================================================================
# PyPowSyBl-specific processing functions
# ============================================================================


def pypowsybl_post_processing(psy_net: PyPowSyBlNetwork) -> np.ndarray:
    """Post-processes pypowsybl power flow data to match pf_node.csv format.

    Creates a matrix of shape (n_buses, 10) with columns:
    (bus, Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF)

    Args:
        psy_net: PyPowSyBlNetwork container with network and metadata.

    Returns:
        numpy.ndarray: Matrix containing the processed power flow data.
    """
    network = psy_net.network
    n_buses = psy_net.n_buses

    X = np.zeros((n_buses, 10))

    # Get data from pypowsybl
    buses = network.get_buses()
    loads = network.get_loads()
    generators = network.get_generators()

    # Column 0: bus index
    X[:, 0] = np.arange(n_buses)

    # Columns 1-2: Pd, Qd (load demand) - aggregate by bus
    for load_id, load_row in loads.iterrows():
        bus_id = load_row["bus_id"]
        if bus_id in psy_net.bus_id_to_idx:
            bus_idx = psy_net.bus_id_to_idx[bus_id]
            X[bus_idx, 1] += load_row["p"]  # Pd
            X[bus_idx, 2] += load_row["q"]  # Qd

    # Columns 3-4: Pg, Qg (generation) - aggregate by bus
    # Note: pypowsybl uses load convention (negative = generating), so we negate
    gen_buses = set()
    slack_bus_idx = None

    for gen_id, gen_row in generators.iterrows():
        bus_id = gen_row["bus_id"]
        if bus_id in psy_net.bus_id_to_idx:
            bus_idx = psy_net.bus_id_to_idx[bus_id]
            X[bus_idx, 3] += -gen_row["p"]  # Pg (negate)
            X[bus_idx, 4] += -gen_row["q"]  # Qg (negate)
            gen_buses.add(bus_idx)

            # First generator with voltage regulator is likely slack/REF
            if gen_row["voltage_regulator_on"] and slack_bus_idx is None:
                slack_bus_idx = bus_idx

    # If no slack identified, use first generator bus
    if slack_bus_idx is None and gen_buses:
        slack_bus_idx = min(gen_buses)

    # Columns 5-6: Vm (p.u.), Va (degrees)
    for bus_id, bus_row in buses.iterrows():
        if bus_id in psy_net.bus_id_to_idx:
            bus_idx = psy_net.bus_id_to_idx[bus_id]
            nominal_v = psy_net.nominal_voltages[bus_id]
            X[bus_idx, 5] = bus_row["v_mag"] / nominal_v  # Vm in p.u.
            X[bus_idx, 6] = bus_row["v_angle"]  # Va in degrees

    # Columns 7-9: PQ, PV, REF (one-hot encoded bus types)
    for bus_idx in range(n_buses):
        if bus_idx == slack_bus_idx:
            X[bus_idx, 9] = 1.0  # REF
        elif bus_idx in gen_buses:
            X[bus_idx, 8] = 1.0  # PV
        else:
            X[bus_idx, 7] = 1.0  # PQ

    return X


def get_adjacency_list_pypowsybl(psy_net: PyPowSyBlNetwork) -> np.ndarray:
    """Gets adjacency list (Y-bus) for pypowsybl network.

    Builds Y-bus matrix natively from pypowsybl network parameters.
    Uses pypowsybl's per_unit mode to get values on 100 MVA base.

    Args:
        psy_net: PyPowSyBlNetwork container with network and metadata.

    Returns:
        numpy.ndarray: Array containing edge indices and attributes (G, B).
    """
    network = psy_net.network
    n_buses = psy_net.n_buses

    # Enable per-unit mode for correct parameter scaling
    original_per_unit = network.per_unit
    network.per_unit = True

    try:
        # Initialize Y-bus as dense complex matrix
        Y_bus = np.zeros((n_buses, n_buses), dtype=complex)

        # Get voltage levels for tap ratio calculation
        vls = network.get_voltage_levels()

        # Process lines (values now in per-unit)
        lines = network.get_lines()
        for line_id, line in lines.iterrows():
            bus1_id = line["bus1_id"]
            bus2_id = line["bus2_id"]

            if bus1_id not in psy_net.bus_id_to_idx or bus2_id not in psy_net.bus_id_to_idx:
                continue

            i = psy_net.bus_id_to_idx[bus1_id]
            j = psy_net.bus_id_to_idx[bus2_id]

            # In per-unit mode, r and x are already in p.u.
            r_pu = line["r"]
            x_pu = line["x"]

            # Series admittance
            z_pu = complex(r_pu, x_pu)
            if abs(z_pu) > 1e-12:
                y_series = 1.0 / z_pu
            else:
                y_series = 0.0

            # Shunt admittance (b1, b2 already in p.u. in per_unit mode)
            b1_pu = line["b1"] if not np.isnan(line["b1"]) else 0.0
            b2_pu = line["b2"] if not np.isnan(line["b2"]) else 0.0
            y_shunt_1 = complex(0, b1_pu)
            y_shunt_2 = complex(0, b2_pu)

            # Add to Y-bus (pi-model)
            Y_bus[i, i] += y_series + y_shunt_1
            Y_bus[j, j] += y_series + y_shunt_2
            Y_bus[i, j] -= y_series
            Y_bus[j, i] -= y_series

        # Process 2-winding transformers (values now in per-unit)
        trafos = network.get_2_windings_transformers()
        for trafo_id, trafo in trafos.iterrows():
            bus1_id = trafo["bus1_id"]
            bus2_id = trafo["bus2_id"]

            if bus1_id not in psy_net.bus_id_to_idx or bus2_id not in psy_net.bus_id_to_idx:
                continue

            i = psy_net.bus_id_to_idx[bus1_id]  # HV side
            j = psy_net.bus_id_to_idx[bus2_id]  # LV side

            # In per-unit mode, r and x are already in p.u.
            r_pu = trafo["r"]
            x_pu = trafo["x"]

            z_pu = complex(r_pu, x_pu)
            if abs(z_pu) > 1e-12:
                y_series = 1.0 / z_pu
            else:
                y_series = 0.0

            # Turns ratio (off-nominal tap)
            vl1_id = trafo["voltage_level1_id"]
            vl2_id = trafo["voltage_level2_id"]
            V_nom_1 = vls.loc[vl1_id, "nominal_v"]  # kV
            V_nom_2 = vls.loc[vl2_id, "nominal_v"]  # kV
            rated_u1 = trafo["rated_u1"]  # kV
            rated_u2 = trafo["rated_u2"]  # kV

            # Off-nominal tap ratio
            a_nom = V_nom_1 / V_nom_2 if V_nom_2 != 0 else 1.0
            a_actual = rated_u1 / rated_u2 if rated_u2 != 0 else 1.0
            tap = a_actual / a_nom if a_nom != 0 else 1.0

            # Transformer pi-model with tap on HV side
            tap2 = tap * tap
            Y_bus[i, i] += y_series / tap2
            Y_bus[j, j] += y_series
            Y_bus[i, j] -= y_series / tap
            Y_bus[j, i] -= y_series / tap

    finally:
        # Restore original per_unit setting
        network.per_unit = original_per_unit

    # Convert to adjacency list format
    i_idx, j_idx = np.nonzero(Y_bus)
    s = Y_bus[i_idx, j_idx]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i_idx, j_idx))
    edge_attr = np.stack((G, B)).T
    adjacency_lists = np.column_stack((edge_index, edge_attr))
    return adjacency_lists


def process_scenario_pypowsybl(
    psy_net: PyPowSyBlNetwork,
    scenarios: np.ndarray,
    scenario_index: int,
    local_csv_data: List[np.ndarray],
    local_adjacency_lists: List[np.ndarray],
    error_log_file: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Processes a load scenario using pypowsybl native solver.

    Args:
        psy_net: PyPowSyBlNetwork container with network and metadata.
        scenarios: Array of load scenarios of shape (n_loads, n_scenarios, 2).
        scenario_index: Index of the current scenario.
        local_csv_data: List to store processed CSV data.
        local_adjacency_lists: List to store adjacency lists.
        error_log_file: Path to error log file.

    Returns:
        Tuple containing:
            - List of processed CSV data
            - List of adjacency lists
    """
    import pandas as pd

    # Apply the load scenario to the network
    load_df = pd.DataFrame({
        "id": psy_net.load_ids,
        "p0": scenarios[:, scenario_index, 0],
        "q0": scenarios[:, scenario_index, 1],
    })
    load_df = load_df.set_index("id")
    psy_net.network.update_loads(load_df)

    try:
        # Run AC power flow
        converged = run_pf_pypowsybl(psy_net)
        if not converged:
            with open(error_log_file, "a") as f:
                f.write(f"Power flow did not converge at scenario {scenario_index}\n")
            return local_csv_data, local_adjacency_lists

    except Exception as e:
        with open(error_log_file, "a") as f:
            f.write(
                f"Caught an exception at scenario {scenario_index} in run_pf_pypowsybl: {e}\n",
            )
        return local_csv_data, local_adjacency_lists

    # Post-process results
    local_csv_data.extend(pypowsybl_post_processing(psy_net))
    local_adjacency_lists.append(get_adjacency_list_pypowsybl(psy_net))

    return local_csv_data, local_adjacency_lists


def process_scenario_chunk_pypowsybl(
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    psy_net: PyPowSyBlNetwork,
    progress_queue: Queue,
    error_log_path: str,
) -> Tuple[
    Union[None, Exception],
    Union[None, str],
    List[np.ndarray],
    List[np.ndarray],
]:
    """Process scenarios for pypowsybl networks (no topology/generation perturbation).

    Args:
        start_idx: Starting scenario index.
        end_idx: Ending scenario index (exclusive).
        scenarios: Array of load scenarios.
        psy_net: PyPowSyBlNetwork container.
        progress_queue: Queue for progress updates.
        error_log_path: Path to error log file.

    Returns:
        Tuple containing:
            - Exception if error occurred, None otherwise
            - Traceback string if error, None otherwise
            - List of processed CSV data
            - List of adjacency lists
    """
    try:
        local_csv_data = []
        local_adjacency_lists = []

        for scenario_index in range(start_idx, end_idx):
            local_csv_data, local_adjacency_lists = process_scenario_pypowsybl(
                psy_net,
                scenarios,
                scenario_index,
                local_csv_data,
                local_adjacency_lists,
                error_log_path,
            )
            progress_queue.put(1)

        return None, None, local_csv_data, local_adjacency_lists

    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Caught an exception in process_scenario_chunk_pypowsybl: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        for _ in range(end_idx - start_idx):
            progress_queue.put(1)
        return e, traceback.format_exc(), None, None


def process_scenario_contingency(
    net: pandapowerNet,
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
    net = copy.deepcopy(net)

    # apply the load scenario to the network
    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]

    # first run OPF to get the gen set points
    try:
        run_opf(net)
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

    net_pf = copy.deepcopy(net)
    net_pf = pf_preprocessing(net_pf)

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net_pf)

    # Apply generation perturbations
    perturbations = generation_generator.generate(perturbations)

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    # to simulate contingency, we apply the topology perturbation after OPF
    for perturbation in perturbations:
        try:
            # run DCPF for benchmarking purposes
            pp.rundcpp(perturbation)
            perturbation.bus["Vm_dc"] = perturbation.res_bus.vm_pu
            perturbation.bus["Va_dc"] = perturbation.res_bus.va_degree
            # run AC-PF to get the new state of the network after contingency (we don't model any remedial actions)
            run_pf(perturbation)
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
        local_branch_idx_removed.append(
            get_branch_idx_removed(perturbation._ppc["branch"]),
        )
        if not no_stats:
            local_stats.update(perturbation)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats


def process_scenario_chunk(
    mode,
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    net: pandapowerNet,
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
    net: pandapowerNet,
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
    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net)

    # Apply generation perturbations
    perturbations = generation_generator.generate(perturbations)

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    for perturbation in perturbations:
        try:
            # run OPF to get the gen set points. Here the set points account for the topology perturbation.
            run_opf(perturbation)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
                )
            continue

        net_pf = copy.deepcopy(perturbation)

        net_pf = pf_preprocessing(net_pf)

        try:
            # This is not striclty necessary as we havent't changed the setpoints nor the load since we solved OPF, but it gives more accurate PF results
            run_pf(net_pf)

        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_pf function: {e}\n",
                )
            continue

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
        local_branch_idx_removed.append(get_branch_idx_removed(net_pf._ppc["branch"]))
        if not no_stats:
            local_stats.update(net_pf)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats
