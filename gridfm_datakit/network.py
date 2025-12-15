import os
import tempfile
from typing import Union, Dict, Any
from dataclasses import dataclass
from pandapower.auxiliary import pandapowerNet
import requests
from importlib import resources
import pandapower as pp
import pypowsybl.network as ppsn
import pypowsybl as ppsy
import pandapower.networks as pn
import pandas as pd
import warnings


def load_net_from_pp(grid_name: str) -> pandapowerNet:
    """Loads a network from the pandapower library.

    Args:
        grid_name: Name of the grid case file in pandapower library.

    Returns:
        pandapowerNet: Loaded power network configuration.
    """
    network = getattr(pn, grid_name)()
    return network


def load_net_from_file(network_path: str) -> pandapowerNet:
    """Loads a network from a matpower file.

    Args:
        network_path: Path to the matpower file (without extension).

    Returns:
        pandapowerNet: Loaded power network configuration with reindexed buses.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    network = pp.converter.from_mpc(str(network_path))
    warnings.resetwarnings()

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network


def load_net_from_pglib(grid_name: str) -> pandapowerNet:
    """Loads a power grid network from PGLib.

    Downloads the network file if not locally available and loads it into a pandapower network.
    The buses are reindexed to ensure continuous indices.

    Args:
        grid_name: Name of the grid file without the prefix 'pglib_opf_' (e.g., 'case14_ieee', 'case118_ieee').

    Returns:
        pandapowerNet: Loaded power network configuration with reindexed buses.

    Raises:
        requests.exceptions.RequestException: If download fails.
    """
    # Construct file paths
    file_path = str(
        resources.files("gridfm_datakit.grids").joinpath(f"pglib_opf_{grid_name}.m"),
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Download file if not exists
    if not os.path.exists(file_path):
        url = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_{grid_name}.m"
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

    # Load network from file
    warnings.filterwarnings("ignore", category=FutureWarning)
    network = pp.converter.from_mpc(file_path)
    warnings.resetwarnings()

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network

_PYPOWSYBL_GRID_MAP = {
    "case9_ieee": ppsn.create_ieee9,
    "case14_ieee": ppsn.create_ieee14,
    "case30_ieee": ppsn.create_ieee30,
    "case57_ieee": ppsn.create_ieee57,
    "case118_ieee": ppsn.create_ieee118,
    "case300_ieee": ppsn.create_ieee300,
}


@dataclass
class PyPowSyBlNetwork:
    """Container for pypowsybl network with metadata needed for processing.

    Attributes:
        network: The native pypowsybl Network object.
        bus_id_to_idx: Mapping from pypowsybl bus IDs (str) to integer indices.
        idx_to_bus_id: Mapping from integer indices to pypowsybl bus IDs.
        load_ids: List of load IDs in order matching scenario arrays.
        nominal_voltages: Dict mapping bus ID to nominal voltage in kV.
        n_buses: Number of buses in the network.
        n_loads: Number of loads in the network.
    """
    network: ppsy.network.Network
    bus_id_to_idx: Dict[str, int]
    idx_to_bus_id: Dict[int, str]
    load_ids: list
    nominal_voltages: Dict[str, float]
    n_buses: int
    n_loads: int


def _normalize_network_for_opf(net: pandapowerNet) -> None:
    """Normalizes a network to be OPF-compatible by setting missing constraints.

    PyPowSyBl IEEE exports lack OPF-specific data like bus voltage limits,
    generator power limits, and cost functions. This function adds sensible
    defaults to make the network usable with pandapower's OPF solver.

    The heuristics estimate thermal limits based on line impedances and
    typical power system loading factors.

    Args:
        net: pandapower network to normalize (modified in-place).
    """
    import numpy as np

    # Set bus voltage limits if missing/invalid (0.0 means unset)
    default_min_vm = 0.94
    default_max_vm = 1.06

    if (net.bus["min_vm_pu"] == 0.0).any() or net.bus["min_vm_pu"].isna().any():
        net.bus["min_vm_pu"] = default_min_vm

    if (net.bus["max_vm_pu"] == 0.0).any() or net.bus["max_vm_pu"].isna().any():
        net.bus["max_vm_pu"] = default_max_vm

    # Normalize generator voltage setpoints to 1.0 pu (standard practice)
    net.gen["vm_pu"] = 1.0
    net.ext_grid["vm_pu"] = 1.0

    # Set realistic generator power limits based on initial p_mw
    for idx in net.gen.index:
        p_mw = net.gen.loc[idx, "p_mw"]
        if net.gen.loc[idx, "min_p_mw"] < -1000:
            net.gen.loc[idx, "min_p_mw"] = 0.0
        if net.gen.loc[idx, "max_p_mw"] > 1000:
            # Set max to 2x initial or at least 100 MW
            net.gen.loc[idx, "max_p_mw"] = max(p_mw * 2, 100.0) if p_mw > 0 else 0.0

    # Set ext_grid limits if unbounded
    for idx in net.ext_grid.index:
        if net.ext_grid.loc[idx, "min_p_mw"] < -1000:
            net.ext_grid.loc[idx, "min_p_mw"] = 0.0
        if net.ext_grid.loc[idx, "max_p_mw"] > 1000:
            # Estimate based on total load
            total_load = net.load["p_mw"].sum()
            net.ext_grid.loc[idx, "max_p_mw"] = total_load * 1.5
        if net.ext_grid.loc[idx, "min_q_mvar"] < -1000:
            net.ext_grid.loc[idx, "min_q_mvar"] = -net.load["q_mvar"].sum()
        if net.ext_grid.loc[idx, "max_q_mvar"] > 1000:
            net.ext_grid.loc[idx, "max_q_mvar"] = net.load["q_mvar"].sum()

    # Add cost functions if missing (minimize total generation)
    if len(net.poly_cost) == 0:
        # Add cost for ext_grid
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, "ext_grid", cp1_eur_per_mw=1.0)
        # Add cost for generators
        for idx in net.gen.index:
            pp.create_poly_cost(net, idx, "gen", cp1_eur_per_mw=1.0)

    # Calculate realistic line ratings based on impedance
    # Using thermal limit estimation: I_max ≈ k * V_base / Z
    # where k is a scaling factor for typical loading
    for idx in net.line.index:
        if net.line.loc[idx, "max_i_ka"] > 10000:
            # Get line parameters
            from_bus = int(net.line.loc[idx, "from_bus"])
            vn_kv = net.bus.loc[from_bus, "vn_kv"]
            r_ohm = net.line.loc[idx, "r_ohm_per_km"] * net.line.loc[idx, "length_km"]
            x_ohm = net.line.loc[idx, "x_ohm_per_km"] * net.line.loc[idx, "length_km"]
            z_ohm = np.sqrt(r_ohm**2 + x_ohm**2)

            if z_ohm > 0:
                # Estimate current rating: typical loading ~70% of thermal limit
                # S_max ≈ V² / Z, I_max ≈ S_max / (√3 * V)
                s_max_mva = (vn_kv**2 / z_ohm) * 0.3  # 30% of theoretical max
                i_max_ka = s_max_mva / (np.sqrt(3) * vn_kv)
                net.line.loc[idx, "max_i_ka"] = max(i_max_ka, 0.3)  # Min 0.3 kA
            else:
                net.line.loc[idx, "max_i_ka"] = 1.0  # Default for zero impedance

    # Fix transformer ratings and impedance parameters
    # MATPOWER converter assigns sn_mva=99999 when not specified, which causes
    # vk_percent to be 1000x too high (vk_percent = x_pu * 100 * sn_mva/baseMVA)
    # Fix: rescale sn_mva to baseMVA (100) and recalculate vk_percent
    base_mva = 100.0  # Standard MATPOWER baseMVA
    for idx in net.trafo.index:
        old_sn = net.trafo.loc[idx, "sn_mva"]
        if old_sn > 10000:
            # Rescale vk_percent to maintain same impedance with new sn_mva
            # vk_new = vk_old * (sn_new / sn_old)
            old_vk = net.trafo.loc[idx, "vk_percent"]
            new_sn = base_mva
            new_vk = old_vk * (new_sn / old_sn)

            net.trafo.loc[idx, "sn_mva"] = new_sn
            net.trafo.loc[idx, "vk_percent"] = new_vk

            # Also rescale vkr_percent if non-zero
            old_vkr = net.trafo.loc[idx, "vkr_percent"]
            if old_vkr != 0:
                net.trafo.loc[idx, "vkr_percent"] = old_vkr * (new_sn / old_sn)


def load_net_from_pypowsybl(grid_name: str) -> PyPowSyBlNetwork:
    """Loads a network from PyPowSyBl as a native pypowsybl Network.

    Creates a PyPowSyBl IEEE test case and returns it wrapped with metadata
    needed for power flow processing and output generation.

    Args:
        grid_name: Name of the grid (e.g., 'case30_ieee', 'case118_ieee').

    Returns:
        PyPowSyBlNetwork: Container with native network and processing metadata.

    Raises:
        ValueError: If grid_name is not a supported IEEE test case.
    """
    if grid_name not in _PYPOWSYBL_GRID_MAP:
        raise ValueError(
            f"Invalid grid name: {grid_name}. "
            f"Supported grids: {list(_PYPOWSYBL_GRID_MAP.keys())}"
        )

    # Create pypowsybl network
    network = _PYPOWSYBL_GRID_MAP[grid_name]()

    # Build bus ID to index mapping
    buses = network.get_buses()
    bus_ids = list(buses.index)
    bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    idx_to_bus_id = {idx: bus_id for bus_id, idx in bus_id_to_idx.items()}

    # Get load IDs in consistent order
    loads = network.get_loads()
    load_ids = list(loads.index)

    # Build nominal voltage mapping (bus_id -> kV)
    voltage_levels = network.get_voltage_levels()
    buses_df = network.get_buses()
    nominal_voltages = {}
    for bus_id in bus_ids:
        vl_id = buses_df.loc[bus_id, "voltage_level_id"]
        nominal_voltages[bus_id] = voltage_levels.loc[vl_id, "nominal_v"]

    return PyPowSyBlNetwork(
        network=network,
        bus_id_to_idx=bus_id_to_idx,
        idx_to_bus_id=idx_to_bus_id,
        load_ids=load_ids,
        nominal_voltages=nominal_voltages,
        n_buses=len(bus_ids),
        n_loads=len(load_ids),
    )