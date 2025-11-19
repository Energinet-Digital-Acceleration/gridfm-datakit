import pandapower as pp
import numpy as np
import pandas as pd
from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
import os
from pandapower.pypower.idx_brch import T_BUS, F_BUS, RATE_A
from pandapower.pypower.makeYbus import branch_vectors
from typing import List, Union
from gridfm_datakit.network_interface import NetworkInterface
from gridfm_datakit.process.solver_interface import SolverInterface


def save_edge_params(net: Union[pandapowerNet, Network], path: str):
    """Saves edge parameters for the network to a CSV file.

    Extracts and saves branch parameters including admittance matrices and rate limits.

    Args:
        net: The power network (pandapower or pypowsybl).
        path: Path where the edge parameters CSV file should be saved.
    """
    # Run DCPF to create ppc structure if needed
    if isinstance(net, pandapowerNet):
        pp.rundcpp(net)
        ppc = net._ppc
    else:  # pypowsybl
        # Run DC power flow to populate results
        solver = SolverInterface.create_solver(net)
        solver.run_dcpf()

        # Get ppc structure via adapter
        adapter = NetworkInterface.create_adapter(net)
        ppc = adapter.get_ppc()
    to_bus = np.real(ppc["branch"][:, T_BUS])
    from_bus = np.real(ppc["branch"][:, F_BUS])

    # Calculate branch admittances
    if isinstance(net, pandapowerNet):
        # For pandapower, use branch_vectors which handles asymmetric branches
        Ytt, Yff, Yft, Ytf = branch_vectors(ppc["branch"], ppc["branch"].shape[0])
    else:
        # For pypowsybl, calculate admittances directly from ppc branch data
        # ppc["branch"] format: [fbus, tbus, r, x, b, rate_a, rate_b, rate_c, tap, shift, status]
        from pandapower.pypower.idx_brch import BR_R, BR_X, BR_B

        r = ppc["branch"][:, BR_R]
        x = ppc["branch"][:, BR_X]
        b = ppc["branch"][:, BR_B]

        # Series admittance
        y_series = 1.0 / (r + 1j * x)

        # Shunt admittance (split equally)
        y_shunt = 1j * b / 2.0

        # Branch admittances (simplified pi-model)
        Yff = y_series + y_shunt
        Yft = -y_series
        Ytf = -y_series
        Ytt = y_series + y_shunt

    Ytt_r = np.real(Ytt)
    Ytt_i = np.imag(Ytt)
    Yff_r = np.real(Yff)
    Yff_i = np.imag(Yff)
    Yft_r = np.real(Yft)
    Yft_i = np.imag(Yft)
    Ytf_r = np.real(Ytf)
    Ytf_i = np.imag(Ytf)

    rate_a = np.real(ppc["branch"][:, RATE_A])
    edge_params = pd.DataFrame(
        np.column_stack(
            (
                from_bus,
                to_bus,
                Yff_r,
                Yff_i,
                Yft_r,
                Yft_i,
                Ytf_r,
                Ytf_i,
                Ytt_r,
                Ytt_i,
                rate_a,
            ),
        ),
        columns=[
            "from_bus",
            "to_bus",
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
            "Ytf_r",
            "Ytf_i",
            "Ytt_r",
            "Ytt_i",
            "rate_a",
        ],
    )
    # comvert everything to float32
    edge_params = edge_params.astype(np.float32)
    edge_params.to_csv(path, index=False)


def save_bus_params(net: Union[pandapowerNet, Network], path: str):
    """Saves bus parameters for the network to a CSV file.

    Extracts and saves bus parameters including voltage limits and base values.

    Args:
        net: The power network (pandapower or pypowsybl).
        path: Path where the bus parameters CSV file should be saved.
    """
    adapter = NetworkInterface.create_adapter(net)
    buses = adapter.get_buses()

    idx = buses.index
    bus_type = buses["type"]

    if isinstance(net, pandapowerNet):
        base_kv = buses["vn_kv"]
        vmin = buses["min_vm_pu"]
        vmax = buses["max_vm_pu"]
    else:  # pypowsybl
        # Get voltage levels for base voltage
        vl = net.get_voltage_levels()
        # Map buses to voltage levels
        bus_vl_map = buses["voltage_level_id"].map(vl["nominal_v"])
        base_kv = bus_vl_map.values

        # Get voltage limits
        vmin_map = buses["voltage_level_id"].map(vl["low_voltage_limit"])
        vmax_map = buses["voltage_level_id"].map(vl["high_voltage_limit"])
        vmin = (vmin_map / base_kv).fillna(0.9).values
        vmax = (vmax_map / base_kv).fillna(1.1).values

    bus_params = pd.DataFrame(
        np.column_stack((idx, bus_type, vmin, vmax, base_kv)),
        columns=["bus", "type", "vmin", "vmax", "baseKV"],
    )
    bus_params.to_csv(path, index=False)


def save_branch_idx_removed(branch_idx_removed: List[List[int]], path: str):
    """Saves indices of removed branches for each scenario.

    Appends the removed branch indices to an existing CSV file or creates a new one.

    Args:
        branch_idx_removed: List of removed branch indices for each scenario.
        path: Path where the branch indices CSV file should be saved.
    """
    if os.path.exists(path):
        existing_df = pd.read_csv(path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]
    else:
        last_scenario = -1

    scenario_idx = np.arange(
        last_scenario + 1,
        last_scenario + 1 + len(branch_idx_removed),
    )
    branch_idx_removed_df = pd.DataFrame(branch_idx_removed)
    branch_idx_removed_df.insert(0, "scenario", scenario_idx)
    branch_idx_removed_df.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )  # append to existing file or create new one


def save_node_edge_data(
    net: Union[pandapowerNet, Network],
    node_path: str,
    edge_path: str,
    csv_data: list,
    adjacency_lists: list,
    mode: str = "pf",
):
    """Saves generated node and edge data to CSV files.

    Saves generated data for nodes and edges,
    appending to existing files if they exist.

    Args:
        net: The power network (pandapower or pypowsybl).
        node_path: Path where node data should be saved.
        edge_path: Path where edge data should be saved.
        csv_data: List of node-level data for each scenario.
        adjacency_lists: List of edge-level adjacency lists for each scenario.
        mode: Analysis mode, either 'pf' for power flow or 'contingency' for contingency analysis.
    """
    adapter = NetworkInterface.create_adapter(net)
    n_buses = adapter.get_num_buses()

    # Determine last scenario index
    last_scenario = -1
    if os.path.exists(node_path):
        existing_df = pd.read_csv(node_path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]

    # Create DataFrame for node data
    if mode == "pf":
        df = pd.DataFrame(
            csv_data,
            columns=[
                "bus",
                "Pd",
                "Qd",
                "Pg",
                "Qg",
                "Vm",
                "Va",
                "PQ",
                "PV",
                "REF",
            ],
        )
    elif (
        mode == "contingency"
    ):  # we add the dc voltage to the node data for benchmarking purposes
        df = pd.DataFrame(
            csv_data,
            columns=[
                "bus",
                "Pd",
                "Qd",
                "Pg",
                "Qg",
                "Vm",
                "Va",
                "PQ",
                "PV",
                "REF",
                "Vm_dc",
                "Va_dc",
            ],
        )

    df["bus"] = df["bus"].astype("int64")

    # Shift scenario indices
    scenario_indices = np.repeat(
        range(last_scenario + 1, last_scenario + 1 + (df.shape[0] // n_buses)),
        n_buses,
    )  # repeat each scenario index n_buses times since there are n_buses rows for each scenario
    df.insert(0, "scenario", scenario_indices)

    # Append to CSV
    df.to_csv(node_path, mode="a", header=not os.path.exists(node_path), index=False)

    # Create DataFrame for edge data
    adj_df = pd.DataFrame(
        np.concatenate(adjacency_lists),
        columns=["index1", "index2", "G", "B"],
    )

    # Convert index columns to integers first, then format as strings with .0
    adj_df["index1"] = adj_df["index1"].astype("int64").apply(lambda x: f"{x}.0")
    adj_df["index2"] = adj_df["index2"].astype("int64").apply(lambda x: f"{x}.0")

    # Format G column with 16 decimals but remove trailing zeros
    def format_float_no_trailing_zeros(x, decimals):
        s = f"{x:.{decimals}f}"
        s = s.rstrip('0').rstrip('.')
        if '.' not in s:
            s += '.0'
        return s

    adj_df["G"] = adj_df["G"].astype("float64").apply(lambda x: format_float_no_trailing_zeros(x, 16))
    adj_df["B"] = adj_df["B"].astype("float64").apply(lambda x: format_float_no_trailing_zeros(x, 16))

    # Shift scenario indices
    scenario_indices = np.concatenate(
        [
            np.full(adjacency_lists[i].shape[0], last_scenario + 1 + i, dtype="int64")
            for i in range(len(adjacency_lists))
        ],
    )  # for each scenario, we repeat the scenario index as many times as there are edges in the scenario
    adj_df.insert(0, "scenario", scenario_indices)

    # Append to CSV
    adj_df.to_csv(
        edge_path,
        mode="a",
        header=not os.path.exists(edge_path),
        index=False,
    )
