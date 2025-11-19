"""Unified interface for power network representations.

Provides abstract base class and adapters for both pandapower and pypowsybl networks.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import copy
from typing import Union, Tuple, Dict, Any
from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
import pandapower as pp
from pandapower import makeYbus_pypower
import pypowsybl as pps


def copy_network(net: Union[pandapowerNet, Network]) -> Union[pandapowerNet, Network]:
    """Create a deep copy of a network, preserving custom attributes for pypowsybl.

    Args:
        net: Network to copy (pandapower or pypowsybl)

    Returns:
        Deep copy of the network
    """
    if isinstance(net, Network):
        # For pypowsybl, we need to handle custom attributes specially
        # Save custom attributes before copy
        custom_attrs = None
        if hasattr(net, '_gridfm_custom_attrs'):
            custom_attrs = copy.deepcopy(net._gridfm_custom_attrs)

        # Create deep copy of network
        net_copy = copy.deepcopy(net)

        # Restore custom attributes
        if custom_attrs is not None:
            net_copy._gridfm_custom_attrs = custom_attrs
        elif not hasattr(net_copy, '_gridfm_custom_attrs'):
            net_copy._gridfm_custom_attrs = {
                'bus': {}, 'load': {}, 'gen': {}, 'line': {}, 'trafo': {}, 'shunt': {}
            }

        return net_copy
    else:
        # For pandapower, regular deepcopy works fine
        return copy.deepcopy(net)


class NetworkInterface(ABC):
    """Abstract interface for power network operations."""

    def __init__(self, network: Union[pandapowerNet, Network]):
        self._network = network

    @property
    def network(self) -> Union[pandapowerNet, Network]:
        """Get underlying network object."""
        return self._network

    @staticmethod
    def create_adapter(net: Union[pandapowerNet, Network]) -> "NetworkInterface":
        """Factory method to create appropriate adapter.

        Args:
            net: Either pandapower or pypowsybl network

        Returns:
            Appropriate network adapter
        """
        if isinstance(net, Network):
            return PyPowSyBlNetworkAdapter(net)
        elif isinstance(net, pandapowerNet):
            return PandapowerNetworkAdapter(net)
        else:
            raise TypeError(f"Unsupported network type: {type(net)}")

    # Bus operations
    @abstractmethod
    def get_buses(self) -> pd.DataFrame:
        """Get buses DataFrame."""
        pass

    @abstractmethod
    def update_buses(self, df: pd.DataFrame) -> None:
        """Update buses with modified DataFrame."""
        pass

    @abstractmethod
    def get_num_buses(self) -> int:
        """Get number of buses."""
        pass

    # Load operations
    @abstractmethod
    def get_loads(self) -> pd.DataFrame:
        """Get loads DataFrame."""
        pass

    @abstractmethod
    def update_loads(self, df: pd.DataFrame) -> None:
        """Update loads with modified DataFrame."""
        pass

    # Generator operations
    @abstractmethod
    def get_generators(self) -> pd.DataFrame:
        """Get generators DataFrame (voltage-controlled)."""
        pass

    @abstractmethod
    def get_static_generators(self) -> pd.DataFrame:
        """Get static generators DataFrame (constant P,Q)."""
        pass

    @abstractmethod
    def update_generators(self, df: pd.DataFrame) -> None:
        """Update generators with modified DataFrame."""
        pass

    @abstractmethod
    def update_static_generators(self, df: pd.DataFrame) -> None:
        """Update static generators with modified DataFrame."""
        pass

    # External grid / slack bus
    @abstractmethod
    def get_external_grids(self) -> pd.DataFrame:
        """Get external grids DataFrame."""
        pass

    @abstractmethod
    def identify_slack_bus(self) -> np.ndarray:
        """Identify slack bus(es).

        Returns:
            Array of slack bus indices/IDs
        """
        pass

    # Line operations
    @abstractmethod
    def get_lines(self) -> pd.DataFrame:
        """Get lines DataFrame."""
        pass

    @abstractmethod
    def update_lines(self, df: pd.DataFrame) -> None:
        """Update lines with modified DataFrame."""
        pass

    # Transformer operations
    @abstractmethod
    def get_transformers(self) -> pd.DataFrame:
        """Get transformers DataFrame (2-winding)."""
        pass

    @abstractmethod
    def update_transformers(self, df: pd.DataFrame) -> None:
        """Update transformers with modified DataFrame."""
        pass

    # Shunt operations
    @abstractmethod
    def get_shunts(self) -> pd.DataFrame:
        """Get shunts DataFrame."""
        pass

    @abstractmethod
    def update_shunts(self, df: pd.DataFrame) -> None:
        """Update shunts with modified DataFrame."""
        pass

    # Result access
    @abstractmethod
    def get_result_buses(self) -> pd.DataFrame:
        """Get bus results (voltage, angle)."""
        pass

    @abstractmethod
    def get_result_generators(self) -> pd.DataFrame:
        """Get generator results (P, Q, V)."""
        pass

    @abstractmethod
    def get_result_static_generators(self) -> pd.DataFrame:
        """Get static generator results (P, Q)."""
        pass

    @abstractmethod
    def get_result_external_grids(self) -> pd.DataFrame:
        """Get external grid results (P, Q)."""
        pass

    @abstractmethod
    def get_result_loads(self) -> pd.DataFrame:
        """Get load results (P, Q)."""
        pass

    @abstractmethod
    def get_result_lines(self) -> pd.DataFrame:
        """Get line results (loading, flows)."""
        pass

    @abstractmethod
    def get_result_transformers(self) -> pd.DataFrame:
        """Get transformer results (loading, flows)."""
        pass

    # Network matrix operations
    @abstractmethod
    def get_ybus(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get bus admittance matrix.

        Returns:
            Tuple of (Y_bus, Yf, Yt) matrices
        """
        pass

    @abstractmethod
    def get_ppc(self) -> Dict[str, Any]:
        """Get PYPOWER case format representation.

        Returns:
            Dictionary with 'bus', 'gen', 'branch', 'baseMVA' keys
        """
        pass

    # Base MVA
    @abstractmethod
    def get_base_mva(self) -> float:
        """Get base MVA for per-unit calculations."""
        pass


class PandapowerNetworkAdapter(NetworkInterface):
    """Adapter for pandapower networks."""

    def get_buses(self) -> pd.DataFrame:
        return self._network.bus

    def update_buses(self, df: pd.DataFrame) -> None:
        self._network.bus = df

    def get_num_buses(self) -> int:
        return len(self._network.bus)

    def get_loads(self) -> pd.DataFrame:
        return self._network.load

    def update_loads(self, df: pd.DataFrame) -> None:
        self._network.load = df

    def get_generators(self) -> pd.DataFrame:
        return self._network.gen

    def get_static_generators(self) -> pd.DataFrame:
        return self._network.sgen

    def update_generators(self, df: pd.DataFrame) -> None:
        self._network.gen = df

    def update_static_generators(self, df: pd.DataFrame) -> None:
        self._network.sgen = df

    def get_external_grids(self) -> pd.DataFrame:
        return self._network.ext_grid

    def identify_slack_bus(self) -> np.ndarray:
        return np.unique(np.array(self._network.ext_grid["bus"]))

    def get_lines(self) -> pd.DataFrame:
        return self._network.line

    def update_lines(self, df: pd.DataFrame) -> None:
        self._network.line = df

    def get_transformers(self) -> pd.DataFrame:
        return self._network.trafo

    def update_transformers(self, df: pd.DataFrame) -> None:
        self._network.trafo = df

    def get_shunts(self) -> pd.DataFrame:
        return self._network.shunt

    def update_shunts(self, df: pd.DataFrame) -> None:
        self._network.shunt = df

    def get_result_buses(self) -> pd.DataFrame:
        return self._network.res_bus

    def get_result_generators(self) -> pd.DataFrame:
        return self._network.res_gen

    def get_result_static_generators(self) -> pd.DataFrame:
        return self._network.res_sgen

    def get_result_external_grids(self) -> pd.DataFrame:
        return self._network.res_ext_grid

    def get_result_loads(self) -> pd.DataFrame:
        return self._network.res_load

    def get_result_lines(self) -> pd.DataFrame:
        return self._network.res_line

    def get_result_transformers(self) -> pd.DataFrame:
        return self._network.res_trafo

    def get_ybus(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ppc = self._network._ppc
        return makeYbus_pypower(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    def get_ppc(self) -> Dict[str, Any]:
        return self._network._ppc

    def get_base_mva(self) -> float:
        return self._network.sn_mva


class PyPowSyBlNetworkAdapter(NetworkInterface):
    """Adapter for pypowsybl networks."""

    def __init__(self, network: Network):
        super().__init__(network)
        self._slack_bus_id = None
        # Store custom attributes on the network object itself so they persist across adapters
        if not hasattr(network, '_gridfm_custom_attrs'):
            network._gridfm_custom_attrs = {
                'bus': {},
                'load': {},
                'gen': {},
                'line': {},
                'trafo': {},
                'shunt': {}
            }

    def get_buses(self) -> pd.DataFrame:
        buses = self._network.get_buses()
        # Add custom attributes if they exist
        for attr_name, attr_data in self._network._gridfm_custom_attrs['bus'].items():
            buses[attr_name] = buses.index.map(lambda x: attr_data.get(x))
        return buses

    def update_buses(self, df: pd.DataFrame) -> None:
        # pypowsybl requires specific columns for update
        # Separate pypowsybl-native fields from custom fields
        pypowsybl_columns = {'name', 'v_mag', 'v_angle'}
        custom_columns = set(df.columns) - pypowsybl_columns - {'voltage_level_id', 'connected', 'bus_breaker_section', 'node'}

        # Update pypowsybl-native fields
        for idx, row in df.iterrows():
            kwargs = {'id': idx}
            if 'name' in row and pd.notna(row['name']):
                kwargs['name'] = row['name']
            if 'v_mag' in row and pd.notna(row['v_mag']):
                kwargs['v_mag'] = row['v_mag']
            if 'v_angle' in row and pd.notna(row['v_angle']):
                kwargs['v_angle'] = row['v_angle']
            self._network.update_buses(**kwargs)

        # Store custom attributes
        for col in custom_columns:
            if col not in self._network._gridfm_custom_attrs['bus']:
                self._network._gridfm_custom_attrs['bus'][col] = {}
            for idx, value in df[col].items():
                self._network._gridfm_custom_attrs['bus'][col][idx] = value

    def get_num_buses(self) -> int:
        return len(self._network.get_buses())

    def get_loads(self) -> pd.DataFrame:
        loads = self._network.get_loads()
        # Add custom attributes
        for attr_name, attr_data in self._network._gridfm_custom_attrs['load'].items():
            loads[attr_name] = loads.index.map(lambda x: attr_data.get(x))
        return loads

    def update_loads(self, df: pd.DataFrame) -> None:
        pypowsybl_columns = {'p0', 'q0', 'connected', 'p', 'q', 'bus_id', 'voltage_level_id', 'name'}
        custom_columns = set(df.columns) - pypowsybl_columns

        # Update pypowsybl-native fields
        for idx, row in df.iterrows():
            kwargs = {'id': idx}
            if 'p0' in row and pd.notna(row['p0']):
                kwargs['p0'] = row['p0']
            if 'q0' in row and pd.notna(row['q0']):
                kwargs['q0'] = row['q0']
            if 'connected' in row and pd.notna(row['connected']):
                kwargs['connected'] = row['connected']
            if 'name' in row and pd.notna(row['name']):
                kwargs['name'] = row['name']
            if len(kwargs) > 1:  # More than just 'id'
                self._network.update_loads(**kwargs)

        # Store custom attributes
        for col in custom_columns:
            if col not in self._network._gridfm_custom_attrs['load']:
                self._network._gridfm_custom_attrs['load'][col] = {}
            for idx, value in df[col].items():
                self._network._gridfm_custom_attrs['load'][col][idx] = value

    def get_generators(self) -> pd.DataFrame:
        """Get voltage-controlled generators."""
        gens = self._network.get_generators()
        gens_filtered = gens[gens['voltage_regulator_on'] == True].copy()
        # Add custom attributes
        for attr_name, attr_data in self._network._gridfm_custom_attrs['gen'].items():
            gens_filtered[attr_name] = gens_filtered.index.map(lambda x: attr_data.get(x))
        return gens_filtered

    def get_static_generators(self) -> pd.DataFrame:
        """Get non-voltage-controlled generators (static)."""
        gens = self._network.get_generators()
        gens_filtered = gens[gens['voltage_regulator_on'] == False].copy()
        # Add custom attributes
        for attr_name, attr_data in self._network._gridfm_custom_attrs['gen'].items():
            gens_filtered[attr_name] = gens_filtered.index.map(lambda x: attr_data.get(x))
        return gens_filtered

    def update_generators(self, df: pd.DataFrame) -> None:
        pypowsybl_columns = {'target_p', 'target_v', 'target_q', 'connected', 'bus_id', 'voltage_level_id',
                             'name', 'energy_source', 'min_p', 'max_p', 'min_q', 'max_q',
                             'voltage_regulator_on', 'regulated_element_id', 'p', 'q', 'i', 'v_mag'}
        custom_columns = set(df.columns) - pypowsybl_columns

        # Update pypowsybl-native fields
        for idx, row in df.iterrows():
            kwargs = {'id': idx}
            if 'target_p' in row and pd.notna(row['target_p']):
                kwargs['target_p'] = row['target_p']
            if 'target_v' in row and pd.notna(row['target_v']):
                kwargs['target_v'] = row['target_v']
            if 'target_q' in row and pd.notna(row['target_q']):
                kwargs['target_q'] = row['target_q']
            if 'connected' in row and pd.notna(row['connected']):
                kwargs['connected'] = row['connected']
            if 'name' in row and pd.notna(row['name']):
                kwargs['name'] = row['name']
            if len(kwargs) > 1:
                self._network.update_generators(**kwargs)

        # Store custom attributes
        for col in custom_columns:
            if col not in self._network._gridfm_custom_attrs['gen']:
                self._network._gridfm_custom_attrs['gen'][col] = {}
            for idx, value in df[col].items():
                self._network._gridfm_custom_attrs['gen'][col][idx] = value

    def update_static_generators(self, df: pd.DataFrame) -> None:
        # Same as update_generators for pypowsybl
        self.update_generators(df)

    def get_external_grids(self) -> pd.DataFrame:
        """Return empty DataFrame - pypowsybl doesn't have ext_grid concept."""
        return pd.DataFrame(columns=['name', 'bus'])

    def identify_slack_bus(self) -> np.ndarray:
        """Identify slack bus by running a loadflow.

        Returns:
            Array containing slack bus ID
        """
        if self._slack_bus_id is None:
            # Run AC loadflow to identify slack bus
            try:
                params = pps.loadflow.Parameters(distributed_slack=False)
                result = pps.loadflow.run_ac(self._network, params)
                self._slack_bus_id = result[0].slack_bus_id
            except Exception as e:
                # Fallback: find bus with generator that has highest priority
                # or just use first generator's bus
                gens = self._network.get_generators()
                if len(gens) > 0:
                    self._slack_bus_id = gens.iloc[0]['bus_id']
                else:
                    raise RuntimeError("Cannot identify slack bus and no generators found") from e

        return np.array([self._slack_bus_id])

    def get_lines(self) -> pd.DataFrame:
        return self._network.get_lines()

    def update_lines(self, df: pd.DataFrame) -> None:
        for idx, row in df.iterrows():
            self._network.update_lines(
                id=idx,
                connected1=row.get('connected1') if 'connected1' in row else None,
                connected2=row.get('connected2') if 'connected2' in row else None,
            )

    def get_transformers(self) -> pd.DataFrame:
        return self._network.get_2_windings_transformers()

    def update_transformers(self, df: pd.DataFrame) -> None:
        for idx, row in df.iterrows():
            self._network.update_2_windings_transformers(
                id=idx,
                connected1=row.get('connected1') if 'connected1' in row else None,
                connected2=row.get('connected2') if 'connected2' in row else None,
            )

    def get_shunts(self) -> pd.DataFrame:
        return self._network.get_shunt_compensators()

    def update_shunts(self, df: pd.DataFrame) -> None:
        for idx, row in df.iterrows():
            self._network.update_shunt_compensators(
                id=idx,
                section_count=row.get('section_count') if 'section_count' in row else None,
                connected=row.get('connected') if 'connected' in row else None,
            )

    def get_result_buses(self) -> pd.DataFrame:
        """Get bus results from loadflow.

        Note: pypowsybl stores results in same DataFrame as input data.
        """
        buses = self._network.get_buses()
        # Add bus index and type for compatibility
        buses['bus'] = buses.index
        if 'type' in buses.columns:
            buses['type'] = buses['type']
        return buses

    def get_result_generators(self) -> pd.DataFrame:
        """Get generator results."""
        gens = self._network.get_generators()
        # Filter voltage-controlled
        gens = gens[gens['voltage_regulator_on'] == True].copy()
        # Add bus and type for compatibility
        gens['bus'] = gens['bus_id']
        if 'type' in gens.columns:
            gens['type'] = gens['type']
        return gens

    def get_result_static_generators(self) -> pd.DataFrame:
        """Get static generator results."""
        gens = self._network.get_generators()
        # Filter non-voltage-controlled
        gens = gens[gens['voltage_regulator_on'] == False].copy()
        # Add bus and type for compatibility
        gens['bus'] = gens['bus_id']
        if 'type' in gens.columns:
            gens['type'] = gens['type']
        return gens

    def get_result_external_grids(self) -> pd.DataFrame:
        """Get external grid results - use slack bus generator."""
        slack_bus_id = self.identify_slack_bus()[0]
        gens = self._network.get_generators()
        slack_gens = gens[gens['bus_id'] == slack_bus_id].copy()
        # Add bus column for compatibility
        slack_gens['bus'] = slack_gens['bus_id']
        return slack_gens

    def get_result_loads(self) -> pd.DataFrame:
        """Get load results."""
        loads = self._network.get_loads()
        # Add bus column for compatibility
        loads['bus'] = loads['bus_id']
        return loads

    def get_result_lines(self) -> pd.DataFrame:
        """Get line results."""
        return self._network.get_lines()

    def get_result_transformers(self) -> pd.DataFrame:
        """Get transformer results."""
        return self._network.get_2_windings_transformers()

    def get_ybus(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct Y_bus admittance matrix from pypowsybl network.

        Returns:
            Tuple of (Y_bus, Yf, Yt) matrices
        """
        # Get network components
        buses = self._network.get_buses()
        lines = self._network.get_lines()
        trafos = self._network.get_2_windings_transformers()

        num_buses = len(buses)
        bus_ids = buses.index.tolist()
        bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

        # Initialize matrices
        Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

        # Add line admittances
        for idx, line in lines.iterrows():
            if line.get('connected1', True) and line.get('connected2', True):
                i = bus_id_to_idx[line['bus1_id']]
                j = bus_id_to_idx[line['bus2_id']]

                # Series admittance
                y_series = 1 / complex(line['r'], line['x'])

                # Shunt admittances
                y_shunt = complex(line.get('g1', 0), line.get('b1', 0))
                y_shunt2 = complex(line.get('g2', 0), line.get('b2', 0))

                # Fill Y_bus
                Y_bus[i, i] += y_series + y_shunt
                Y_bus[j, j] += y_series + y_shunt2
                Y_bus[i, j] -= y_series
                Y_bus[j, i] -= y_series

        # Add transformer admittances
        for idx, trafo in trafos.iterrows():
            if trafo.get('connected1', True) and trafo.get('connected2', True):
                i = bus_id_to_idx[trafo['bus1_id']]
                j = bus_id_to_idx[trafo['bus2_id']]

                # Series admittance
                y_series = 1 / complex(trafo['r'], trafo['x'])

                # Shunt admittance
                y_shunt = complex(trafo.get('g', 0), trafo.get('b', 0))

                # Tap ratio
                tap = trafo.get('rated_u2', 1.0) / trafo.get('rated_u1', 1.0)

                # Fill Y_bus (simplified model)
                Y_bus[i, i] += y_series / (tap ** 2) + y_shunt
                Y_bus[j, j] += y_series
                Y_bus[i, j] -= y_series / tap
                Y_bus[j, i] -= y_series / tap

        # For Yf and Yt, we would need from/to incidence matrices
        # Simplified version: return Y_bus three times
        # TODO: Implement proper Yf and Yt if needed
        Yf = Y_bus.copy()
        Yt = Y_bus.copy()

        return Y_bus, Yf, Yt

    def get_ppc(self) -> Dict[str, Any]:
        """Build PYPOWER-compatible case format from pypowsybl network.

        Returns:
            Dictionary with 'bus', 'gen', 'branch', 'baseMVA' keys
        """
        # Get network components
        buses = self._network.get_buses()
        loads = self._network.get_loads()
        gens = self._network.get_generators()
        lines = self._network.get_lines()
        trafos = self._network.get_2_windings_transformers()

        num_buses = len(buses)
        bus_ids = buses.index.tolist()
        bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

        # Build bus matrix (PYPOWER format)
        # Columns: bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin
        bus_matrix = np.zeros((num_buses, 13))

        for idx, (bus_id, bus) in enumerate(buses.iterrows()):
            bus_matrix[idx, 0] = idx  # bus number
            bus_matrix[idx, 1] = 1  # type (PQ for now, will be updated)
            bus_matrix[idx, 7] = bus.get('v_mag', 1.0)  # Vm
            bus_matrix[idx, 8] = bus.get('v_angle', 0.0)  # Va
            bus_matrix[idx, 9] = bus.get('nominal_v', 1.0)  # baseKV
            bus_matrix[idx, 11] = 1.1  # Vmax (default)
            bus_matrix[idx, 12] = 0.9  # Vmin (default)

        # Add loads
        for idx, load in loads.iterrows():
            bus_idx = bus_id_to_idx[load['bus_id']]
            if load.get('connected', True):
                bus_matrix[bus_idx, 2] += load.get('p', load.get('p0', 0))  # Pd
                bus_matrix[bus_idx, 3] += load.get('q', load.get('q0', 0))  # Qd

        # Build gen matrix (PYPOWER format)
        # Columns: bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin
        gen_list = []
        for idx, gen in gens.iterrows():
            if gen.get('connected', True):
                bus_idx = bus_id_to_idx[gen['bus_id']]
                gen_row = np.zeros(10)
                gen_row[0] = bus_idx  # bus
                gen_row[1] = gen.get('p', gen.get('target_p', 0))  # Pg
                gen_row[2] = gen.get('q', gen.get('target_q', 0))  # Qg
                gen_row[3] = gen.get('max_q', 999)  # Qmax
                gen_row[4] = gen.get('min_q', -999)  # Qmin
                gen_row[5] = gen.get('target_v', 1.0)  # Vg
                gen_row[6] = 100  # mBase
                gen_row[7] = 1  # status
                gen_row[8] = gen.get('max_p', 999)  # Pmax
                gen_row[9] = gen.get('min_p', 0)  # Pmin
                gen_list.append(gen_row)

                # Update bus type
                if gen.get('voltage_regulator_on', False):
                    bus_matrix[bus_idx, 1] = 2  # PV bus

        gen_matrix = np.array(gen_list) if gen_list else np.zeros((0, 10))

        # Identify slack bus
        slack_bus_id = self.identify_slack_bus()[0]
        slack_bus_idx = bus_id_to_idx[slack_bus_id]
        bus_matrix[slack_bus_idx, 1] = 3  # REF bus

        # Build branch matrix (PYPOWER format)
        # Columns: fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status
        branch_list = []

        # Add lines
        for idx, line in lines.iterrows():
            if line.get('connected1', True) and line.get('connected2', True):
                branch_row = np.zeros(11)
                branch_row[0] = bus_id_to_idx[line['bus1_id']]  # fbus
                branch_row[1] = bus_id_to_idx[line['bus2_id']]  # tbus
                branch_row[2] = line['r']  # r
                branch_row[3] = line['x']  # x
                branch_row[4] = line.get('b1', 0) + line.get('b2', 0)  # b (total)
                branch_row[5] = line.get('p1', 999)  # rateA
                branch_row[6] = line.get('p1', 999)  # rateB
                branch_row[7] = line.get('p1', 999)  # rateC
                branch_row[8] = 0  # ratio
                branch_row[9] = 0  # angle
                branch_row[10] = 1  # status
                branch_list.append(branch_row)

        # Add transformers
        for idx, trafo in trafos.iterrows():
            if trafo.get('connected1', True) and trafo.get('connected2', True):
                branch_row = np.zeros(11)
                branch_row[0] = bus_id_to_idx[trafo['bus1_id']]  # fbus
                branch_row[1] = bus_id_to_idx[trafo['bus2_id']]  # tbus
                branch_row[2] = trafo['r']  # r
                branch_row[3] = trafo['x']  # x
                branch_row[4] = trafo.get('b', 0)  # b
                branch_row[5] = trafo.get('rated_s', 999)  # rateA
                branch_row[6] = trafo.get('rated_s', 999)  # rateB
                branch_row[7] = trafo.get('rated_s', 999)  # rateC
                branch_row[8] = trafo.get('rated_u2', 1.0) / trafo.get('rated_u1', 1.0)  # ratio
                branch_row[9] = 0  # angle
                branch_row[10] = 1  # status
                branch_list.append(branch_row)

        branch_matrix = np.array(branch_list) if branch_list else np.zeros((0, 11))

        return {
            'baseMVA': 100.0,  # Default base MVA
            'bus': bus_matrix,
            'gen': gen_matrix,
            'branch': branch_matrix,
        }

    def get_base_mva(self) -> float:
        """Get base MVA for per-unit calculations."""
        return 100.0  # Default for pypowsybl
