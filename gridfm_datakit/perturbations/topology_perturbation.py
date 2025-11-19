import numpy as np
import pandapower as pp
from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
import copy
from itertools import combinations
from abc import ABC, abstractmethod
import pandapower.topology as top
import warnings
from typing import Generator, List, Union
from gridfm_datakit.network_interface import NetworkInterface, copy_network
from gridfm_datakit.process.network_utils import check_network_feasibility


# Abstract base class for topology generation
class TopologyGenerator(ABC):
    """Abstract base class for generating perturbed network topologies."""

    def __init__(self) -> None:
        """Initialize the topology generator."""
        pass

    @abstractmethod
    def generate(
        self,
        net: Union[pandapowerNet, Network],
    ) -> Union[Generator[Union[pandapowerNet, Network], None, None], List[Union[pandapowerNet, Network]]]:
        """Generate perturbed topologies.

        Args:
            net: The power network to perturb (pandapower or pypowsybl).

        Yields:
            A perturbed network topology.
        """
        pass


class NoPerturbationGenerator(TopologyGenerator):
    """Generator that yields the original network without any perturbations."""

    def generate(
        self,
        net: Union[pandapowerNet, Network],
    ) -> Generator[Union[pandapowerNet, Network], None, None]:
        """Yield the original network without any perturbations.

        Args:
            net: The power network (pandapower or pypowsybl).

        Yields:
            The original power network.
        """
        yield net


class NMinusKGenerator(TopologyGenerator):
    """Generate perturbed topologies for N-k contingency analysis.

    Only considers lines and transformers. Generates ALL possible topologies with at most k
    components set out of service (lines and transformers).

    Only topologies that are feasible (= no unsupplied buses) are yielded.

    Attributes:
        k: Maximum number of components to drop.
        components_to_drop: List of tuples containing component indices and types.
        component_combinations: List of all possible combinations of components to drop.
    """

    def __init__(self, k: int, base_net: Union[pandapowerNet, Network]) -> None:
        """Initialize the N-k generator.

        Args:
            k: Maximum number of components to drop.
            base_net: The base power network (pandapower or pypowsybl).

        Raises:
            ValueError: If k is 0.
            Warning: If k > 1, as this may result in slow data generation.
        """
        super().__init__()
        if k > 1:
            warnings.warn("k>1. This may result in slow data generation process.")
        if k == 0:
            raise ValueError(
                'k must be greater than 0. Use "none" as argument for the generator_type if you don\'t want to generate any perturbation',
            )
        self.k = k

        # Prepare the list of components to drop using adapter
        adapter = NetworkInterface.create_adapter(base_net)
        lines = adapter.get_lines()
        trafos = adapter.get_transformers()

        self.components_to_drop = [(index, "line") for index in lines.index] + [
            (index, "trafo") for index in trafos.index
        ]

        # Generate all combinations of at most k components
        self.component_combinations = []
        for r in range(self.k + 1):
            self.component_combinations.extend(combinations(self.components_to_drop, r))

        print(
            f"Number of possible topologies with at most {self.k} dropped components: {len(self.component_combinations)}",
        )

    def generate(
        self,
        net: Union[pandapowerNet, Network],
    ) -> Generator[Union[pandapowerNet, Network], None, None]:
        """Generate perturbed topologies by dropping components.

        Args:
            net: The power network (pandapower or pypowsybl).

        Yields:
            A perturbed network topology with at most k components removed.
        """
        for selected_components in self.component_combinations:
            perturbed_topology = copy_network(net)
            adapter = NetworkInterface.create_adapter(perturbed_topology)

            # Separate lines and transformers
            lines_to_drop = [e[0] for e in selected_components if e[1] == "line"]
            trafos_to_drop = [e[0] for e in selected_components if e[1] == "trafo"]

            # Drop selected lines and transformers using adapter
            if lines_to_drop:
                lines = adapter.get_lines()
                if isinstance(net, pandapowerNet):
                    lines.loc[lines_to_drop, "in_service"] = False
                else:  # pypowsybl
                    lines.loc[lines_to_drop, "connected1"] = False
                    lines.loc[lines_to_drop, "connected2"] = False
                adapter.update_lines(lines)

            if trafos_to_drop:
                trafos = adapter.get_transformers()
                if isinstance(net, pandapowerNet):
                    trafos.loc[trafos_to_drop, "in_service"] = False
                else:  # pypowsybl
                    trafos.loc[trafos_to_drop, "connected1"] = False
                    trafos.loc[trafos_to_drop, "connected2"] = False
                adapter.update_transformers(trafos)

            # Check network feasibility and yield the topology
            if check_network_feasibility(perturbed_topology):
                yield perturbed_topology


class RandomComponentDropGenerator(TopologyGenerator):
    """Generate perturbed topologies by randomly setting components out of service.

    Generates perturbed topologies by randomly setting out of service at most k components among the selected element types.
    Only topologies that are feasible (= no unsupplied buses) are yielded.

    Attributes:
        n_topology_variants: Number of topology variants to generate.
        k: Maximum number of components to drop.
        components_to_drop: List of tuples containing component indices and types.
    """

    def __init__(
        self,
        n_topology_variants: int,
        k: int,
        base_net: Union[pandapowerNet, Network],
        elements: List[str] = ["line", "trafo", "gen", "sgen"],
    ) -> None:
        """Initialize the random component drop generator.

        Args:
            n_topology_variants: Number of topology variants to generate.
            k: Maximum number of components to drop.
            base_net: The base power network (pandapower or pypowsybl).
            elements: List of element types to consider for dropping.
        """
        super().__init__()
        self.n_topology_variants = n_topology_variants
        self.k = k

        # Create a list of all components that can be dropped using adapter
        adapter = NetworkInterface.create_adapter(base_net)
        self.components_to_drop = []

        for element in elements:
            if element == "line":
                lines = adapter.get_lines()
                self.components_to_drop.extend(
                    [(index, "line") for index in lines.index]
                )
            elif element == "trafo":
                trafos = adapter.get_transformers()
                self.components_to_drop.extend(
                    [(index, "trafo") for index in trafos.index]
                )
            elif element == "gen":
                gens = adapter.get_generators()
                self.components_to_drop.extend(
                    [(index, "gen") for index in gens.index]
                )
            elif element == "sgen":
                sgens = adapter.get_static_generators()
                self.components_to_drop.extend(
                    [(index, "sgen") for index in sgens.index]
                )

    def generate(
        self,
        net: Union[pandapowerNet, Network],
    ) -> Generator[Union[pandapowerNet, Network], None, None]:
        """Generate perturbed topologies by randomly setting components out of service.

        Args:
            net: The power network (pandapower or pypowsybl).

        Yields:
            A perturbed network topology.
        """
        n_generated_topologies = 0

        # Stop after we generated n_topology_variants
        while n_generated_topologies < self.n_topology_variants:
            perturbed_topology = copy_network(net)
            adapter = NetworkInterface.create_adapter(perturbed_topology)

            # draw the number of components to drop from a uniform distribution
            r = np.random.randint(
                1,
                self.k + 1,
            )  # TODO: decide if we want to be able to set 0 components out of service

            # Randomly select r<=k components to drop
            components = tuple(
                np.random.choice(range(len(self.components_to_drop)), r, replace=False),
            )

            # Convert indices back to actual components
            selected_components = tuple(
                self.components_to_drop[idx] for idx in components
            )

            # Separate lines, transformers, generators, and static generators
            lines_to_drop = [e[0] for e in selected_components if e[1] == "line"]
            trafos_to_drop = [e[0] for e in selected_components if e[1] == "trafo"]
            gens_to_turn_off = [e[0] for e in selected_components if e[1] == "gen"]
            sgens_to_turn_off = [e[0] for e in selected_components if e[1] == "sgen"]

            # Drop selected lines and transformers, turn off generators and static generators
            if lines_to_drop:
                lines = adapter.get_lines()
                if isinstance(net, pandapowerNet):
                    lines.loc[lines_to_drop, "in_service"] = False
                else:  # pypowsybl
                    lines.loc[lines_to_drop, "connected1"] = False
                    lines.loc[lines_to_drop, "connected2"] = False
                adapter.update_lines(lines)

            if trafos_to_drop:
                trafos = adapter.get_transformers()
                if isinstance(net, pandapowerNet):
                    trafos.loc[trafos_to_drop, "in_service"] = False
                else:  # pypowsybl
                    trafos.loc[trafos_to_drop, "connected1"] = False
                    trafos.loc[trafos_to_drop, "connected2"] = False
                adapter.update_transformers(trafos)

            if gens_to_turn_off:
                gens = adapter.get_generators()
                if isinstance(net, pandapowerNet):
                    gens.loc[gens_to_turn_off, "in_service"] = False
                else:  # pypowsybl
                    gens.loc[gens_to_turn_off, "connected"] = False
                adapter.update_generators(gens)

            if sgens_to_turn_off:
                sgens = adapter.get_static_generators()
                if isinstance(net, pandapowerNet):
                    sgens.loc[sgens_to_turn_off, "in_service"] = False
                else:  # pypowsybl
                    sgens.loc[sgens_to_turn_off, "connected"] = False
                adapter.update_static_generators(sgens)

            # Check network feasibility and yield the topology
            if check_network_feasibility(perturbed_topology):
                yield perturbed_topology
                n_generated_topologies += 1
