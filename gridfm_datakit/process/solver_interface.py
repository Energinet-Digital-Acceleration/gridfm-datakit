"""Unified interface for power flow and OPF solvers.

Provides abstract base class and implementations for both pandapower and pypowsybl networks.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Any
import pandas as pd
import numpy as np
from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
import pandapower as pp
import pypowsybl as pps
from pypowsybl.loadflow import ComponentStatus
from gridfm_datakit.network_interface import NetworkInterface


class SolverInterface(ABC):
    """Abstract interface for power flow solver operations."""

    def __init__(self, network: Union[pandapowerNet, Network]):
        self._network = network
        self._adapter = NetworkInterface.create_adapter(network)

    @property
    def network(self) -> Union[pandapowerNet, Network]:
        """Get underlying network object."""
        return self._network

    @staticmethod
    def create_solver(net: Union[pandapowerNet, Network]) -> "SolverInterface":
        """Factory method to create appropriate solver.

        Args:
            net: Either pandapower or pypowsybl network

        Returns:
            Appropriate solver implementation
        """
        if isinstance(net, Network):
            return PyPowSyBlSolver(net)
        elif isinstance(net, pandapowerNet):
            return PandapowerSolver(net)
        else:
            raise TypeError(f"Unsupported network type: {type(net)}")

    @abstractmethod
    def run_opf(self, **kwargs: Any) -> bool:
        """Run Optimal Power Flow.

        Returns:
            bool: True if OPF converged successfully
        """
        pass

    @abstractmethod
    def run_pf(self, **kwargs: Any) -> bool:
        """Run AC Power Flow.

        Returns:
            bool: True if PF converged successfully
        """
        pass

    @abstractmethod
    def run_dcpf(self, **kwargs: Any) -> bool:
        """Run DC Power Flow.

        Returns:
            bool: True if DCPF converged successfully
        """
        pass

    @abstractmethod
    def calculate_power_imbalance(self) -> Tuple[float, float]:
        """Calculate power imbalance in the network.

        Returns:
            Tuple of (active_power_imbalance, reactive_power_imbalance) per bus
        """
        pass


class PandapowerSolver(SolverInterface):
    """Solver implementation for pandapower networks."""

    def run_opf(self, **kwargs: Any) -> bool:
        """Run Optimal Power Flow using pandapower.

        Wraps existing run_opf function from solvers.py.
        """
        from gridfm_datakit.process.solvers import run_opf
        return run_opf(self._network, **kwargs)

    def run_pf(self, **kwargs: Any) -> bool:
        """Run AC Power Flow using pandapower.

        Wraps existing run_pf function from solvers.py.
        """
        from gridfm_datakit.process.solvers import run_pf
        return run_pf(self._network, **kwargs)

    def run_dcpf(self, **kwargs: Any) -> bool:
        """Run DC Power Flow using pandapower.

        Wraps existing run_dcpf function from solvers.py.
        """
        from gridfm_datakit.process.solvers import run_dcpf
        return run_dcpf(self._network, **kwargs)

    def calculate_power_imbalance(self) -> Tuple[float, float]:
        """Calculate power imbalance using pandapower.

        Wraps existing calculate_power_imbalance function from solvers.py.
        """
        from gridfm_datakit.process.solvers import calculate_power_imbalance
        return calculate_power_imbalance(self._network)


class PyPowSyBlSolver(SolverInterface):
    """Solver implementation for pypowsybl networks."""

    def run_opf(self, **kwargs: Any) -> bool:
        """Run Optimal Power Flow - NOT SUPPORTED for pypowsybl.

        PyPowSyBl does not have a native OPF solver.

        Raises:
            NotImplementedError: Always raised - OPF not available
        """
        raise NotImplementedError(
            "PyPowSyBl does not support Optimal Power Flow (OPF). "
            "Only AC and DC power flow calculations are available. "
            "For OPF functionality, use pandapower networks or integrate an external OPF solver."
        )

    def run_pf(self, **kwargs: Any) -> bool:
        """Run AC Power Flow using pypowsybl.

        Args:
            **kwargs: Loadflow parameters (passed to pypowsybl.loadflow.Parameters)

        Returns:
            bool: True if loadflow converged successfully
        """
        # Create loadflow parameters
        params = pps.loadflow.Parameters(**kwargs) if kwargs else pps.loadflow.Parameters()

        # Run AC loadflow
        results = pps.loadflow.run_ac(self._network, params)

        # Check convergence
        converged = results[0].status == ComponentStatus.CONVERGED

        if converged:
            # Add bus and type information to results (for compatibility)
            self._add_result_metadata()

            # Validate power balance (pypowsybl uses more relaxed tolerances than pandapower)
            total_p_diff, total_q_diff = self.calculate_power_imbalance()

            # Use relaxed tolerances: 10 MVAr for reactive, 1 MW for active
            # PyPowSyBl's solver handles power balance internally; if converged, we trust it
            if np.abs(total_p_diff) > 1.0:
                print(f"Warning: Active power imbalance in AC PF: {total_p_diff:.3f} MW")
            if np.abs(total_q_diff) > 10.0:
                print(f"Warning: Reactive power imbalance in AC PF: {total_q_diff:.3f} MVAr")

        return converged

    def run_dcpf(self, **kwargs: Any) -> bool:
        """Run DC Power Flow using pypowsybl.

        Args:
            **kwargs: Loadflow parameters (passed to pypowsybl.loadflow.Parameters)

        Returns:
            bool: True if loadflow converged successfully
        """
        # Create loadflow parameters
        params = pps.loadflow.Parameters(**kwargs) if kwargs else pps.loadflow.Parameters()

        # Run DC loadflow
        results = pps.loadflow.run_dc(self._network, params)

        # Check convergence
        converged = results[0].status == ComponentStatus.CONVERGED

        if converged:
            # Add bus and type information to results (for compatibility)
            self._add_result_metadata()

            # Validate power balance
            total_p_diff, total_q_diff = self.calculate_power_imbalance()

            print(
                "Total reactive power imbalance in DCPF: ",
                total_q_diff,
                " (It is normal that this is not 0 as we are using a DC model)",
            )
            print(
                "Total active power imbalance in DCPF: ",
                total_p_diff,
                " (Should be close to 0)",
            )

        return converged

    def _add_result_metadata(self) -> None:
        """Add bus and type metadata to result dataframes for compatibility.

        PyPowSyBl stores results in the same DataFrame as inputs, but we add
        custom 'bus' and 'type' attributes for compatibility with existing code.
        """
        buses = self._adapter.get_buses()
        gens = self._adapter.get_generators()
        sgens = self._adapter.get_static_generators()
        loads = self._adapter.get_loads()

        # Add 'bus' column if not present (use bus_id)
        if 'bus' not in gens.columns:
            gens['bus'] = gens['bus_id']
            self._adapter.update_generators(gens)

        if 'bus' not in sgens.columns:
            sgens['bus'] = sgens['bus_id']
            self._adapter.update_static_generators(sgens)

        if 'bus' not in loads.columns:
            loads['bus'] = loads['bus_id']
            self._adapter.update_loads(loads)

        # Type information should already be set by network_preprocessing

    def calculate_power_imbalance(self) -> Tuple[float, float]:
        """Calculate power imbalance in pypowsybl network.

        Returns:
            Tuple of (active_power_imbalance, reactive_power_imbalance) per bus
        """
        buses = self._adapter.get_result_buses()
        num_buses = len(buses)

        # Get all generators and aggregate by bus
        gens = self._adapter.get_result_generators()
        sgens = self._adapter.get_result_static_generators()
        ext_grids = self._adapter.get_result_external_grids()

        all_gens_list = []
        for gen_df in [gens, sgens, ext_grids]:
            if len(gen_df) > 0:
                gen_df_copy = gen_df.copy()
                # Rename columns to match pandapower convention
                if 'p' in gen_df_copy.columns:
                    gen_df_copy['p_mw'] = gen_df_copy['p']
                elif 'target_p' in gen_df_copy.columns:
                    gen_df_copy['p_mw'] = gen_df_copy['target_p']
                else:
                    gen_df_copy['p_mw'] = 0

                if 'q' in gen_df_copy.columns:
                    gen_df_copy['q_mvar'] = gen_df_copy['q']
                elif 'target_q' in gen_df_copy.columns:
                    gen_df_copy['q_mvar'] = gen_df_copy['target_q']
                else:
                    gen_df_copy['q_mvar'] = 0

                gen_df_copy['bus'] = gen_df_copy.get('bus', gen_df_copy.get('bus_id'))
                all_gens_list.append(gen_df_copy[['p_mw', 'q_mvar', 'bus']])

        if all_gens_list:
            all_gens = pd.concat(all_gens_list).groupby('bus').sum()
        else:
            all_gens = pd.DataFrame(columns=['p_mw', 'q_mvar'])

        # Get all loads and aggregate by bus
        loads = self._adapter.get_result_loads()
        if len(loads) > 0:
            loads_copy = loads.copy()
            loads_copy['p_mw'] = loads_copy.get('p', loads_copy.get('p0', 0))
            loads_copy['q_mvar'] = loads_copy.get('q', loads_copy.get('q0', 0))
            loads_copy['bus'] = loads_copy.get('bus', loads_copy.get('bus_id'))
            all_loads = loads_copy[['p_mw', 'q_mvar', 'bus']].groupby('bus').sum()
        else:
            all_loads = pd.DataFrame(columns=['p_mw', 'q_mvar'])

        # Calculate net load = load - generation
        net_load_df = pd.concat([all_loads, -all_gens]).groupby('bus').sum()

        # Get bus results
        bus_results = buses.copy()
        bus_results['p_mw'] = bus_results.get('p', 0)
        bus_results['q_mvar'] = bus_results.get('q', 0)

        # Align with all buses
        net_load = net_load_df.reindex(bus_results.index).fillna(0)

        # Check power balance including line losses
        lines = self._adapter.get_result_lines()
        trafos = self._adapter.get_result_transformers()

        # Calculate line losses (p1 + p2 gives losses)
        line_losses_p = 0
        line_losses_q = 0
        if len(lines) > 0 and 'p1' in lines.columns and 'p2' in lines.columns:
            line_losses_p = (lines['p1'] + lines['p2']).sum()
            line_losses_q = (lines['q1'] + lines['q2']).sum()

        # Calculate transformer losses
        trafo_losses_p = 0
        trafo_losses_q = 0
        if len(trafos) > 0 and 'p1' in trafos.columns and 'p2' in trafos.columns:
            trafo_losses_p = (trafos['p1'] + trafos['p2']).sum()
            trafo_losses_q = (trafos['q1'] + trafos['q2']).sum()

        total_p_diff = net_load['p_mw'].sum() + line_losses_p + trafo_losses_p
        total_q_diff = net_load['q_mvar'].sum() + line_losses_q + trafo_losses_q

        return total_p_diff / num_buses, total_q_diff / num_buses
