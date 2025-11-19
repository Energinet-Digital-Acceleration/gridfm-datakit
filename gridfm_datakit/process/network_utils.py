"""Network utility functions for topology validation and analysis.

This module contains utility functions that are used across multiple modules
to avoid circular dependencies.
"""

from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
from typing import Union
from gridfm_datakit.network_interface import NetworkInterface


def check_network_feasibility(net: Union[pandapowerNet, Network]) -> bool:
    """Check if all buses are supplied (connected to slack bus).

    Uses breadth-first search to check if all buses are reachable from the slack bus
    through in-service lines and transformers.

    Args:
        net: The power network (pandapower or pypowsybl).

    Returns:
        bool: True if network is feasible (all buses supplied), False otherwise.
    """
    adapter = NetworkInterface.create_adapter(net)

    # Get slack bus
    slack_buses = adapter.identify_slack_bus()
    if len(slack_buses) == 0:
        return False  # No slack bus

    # Build adjacency graph of in-service connections
    buses = adapter.get_buses()
    lines = adapter.get_lines()
    trafos = adapter.get_transformers()

    # Create bus ID to index mapping for pypowsybl
    if isinstance(net, Network):
        bus_ids = buses.index.tolist()
        bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        slack_idx = bus_id_to_idx[slack_buses[0]]
        num_buses = len(buses)
    else:
        slack_idx = slack_buses[0]
        num_buses = len(buses)

    # Build adjacency list
    adjacency = {i: set() for i in range(num_buses)}

    # Add lines
    for idx, line in lines.iterrows():
        # Check if line is in service
        if isinstance(net, pandapowerNet):
            if not line.get('in_service', True):
                continue
            bus1 = line['from_bus']
            bus2 = line['to_bus']
        else:  # pypowsybl
            if not (line.get('connected1', True) and line.get('connected2', True)):
                continue
            bus1 = bus_id_to_idx[line['bus1_id']]
            bus2 = bus_id_to_idx[line['bus2_id']]

        adjacency[bus1].add(bus2)
        adjacency[bus2].add(bus1)

    # Add transformers
    for idx, trafo in trafos.iterrows():
        # Check if trafo is in service
        if isinstance(net, pandapowerNet):
            if not trafo.get('in_service', True):
                continue
            bus1 = trafo['hv_bus']
            bus2 = trafo['lv_bus']
        else:  # pypowsybl
            if not (trafo.get('connected1', True) and trafo.get('connected2', True)):
                continue
            bus1 = bus_id_to_idx[trafo['bus1_id']]
            bus2 = bus_id_to_idx[trafo['bus2_id']]

        adjacency[bus1].add(bus2)
        adjacency[bus2].add(bus1)

    # BFS from slack bus to find all reachable buses
    visited = set()
    queue = [slack_idx]
    visited.add(slack_idx)

    while queue:
        current = queue.pop(0)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Check if all buses are reachable
    return len(visited) == num_buses
