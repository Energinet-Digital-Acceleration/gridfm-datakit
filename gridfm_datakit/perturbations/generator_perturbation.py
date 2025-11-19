import numpy as np
import pandapower as pp
from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
from abc import ABC, abstractmethod
from typing import Generator, List, Union


class GenerationGenerator(ABC):
    """Abstract base class for applying perturbations to generator elements
    in a network."""

    def __init__(self) -> None:
        """Initialize the generation generator."""
        pass

    @abstractmethod
    def generate(
        self,
        example_generator: Generator[Union[pandapowerNet, Network], None, None],
    ) -> Union[Generator[Union[pandapowerNet, Network], None, None], List[Union[pandapowerNet, Network]]]:
        """Generate generation perturbations.

        Args:
            example_generator: A generator producing example (load/topology/generation)
            scenarios to which line admittance perturbations are added.

        Yields:
            A generation-perturbed scenario.
        """
        pass


class NoGenPerturbationGenerator(GenerationGenerator):
    """Generator that yields the original network generator without any perturbations."""

    def __init__(self):
        """Initialize the no-perturbation generator"""
        pass

    def generate(
        self,
        example_generator: Generator[Union[pandapowerNet, Network], None, None],
    ) -> Generator[Union[pandapowerNet, Network], None, None]:
        """Yield the original examples without any perturbations.

        Args:
            example_generator: A generator producing example
            (load/topology/generation) scenarios to which generator
            cost perturbations should be applied.

        Yields:
            The original example produced by the example_generator.
        """
        for example in example_generator:
            yield example


class PermuteGenCostGenerator(GenerationGenerator):
    """Class for permuting generator costs.

    This class is for generating different generation scenarios
    by permuting all the coeffiecient costs between and among
    generators of power grid networks.
    """

    def __init__(self, base_net: Union[pandapowerNet, Network]) -> None:
        """
        Initialize the gen-cost permuation generator.

        Args:
            base_net: The base power network (pandapower or pypowsybl).

        Raises:
            NotImplementedError: If using pypowsybl network (poly_cost not available).
        """
        if isinstance(base_net, Network):
            raise NotImplementedError(
                "Generator cost perturbations are not supported for pypowsybl networks. "
                "PyPowSyBl does not have poly_cost tables and does not support OPF. "
                "Use 'none' as the generation_perturbation type for pypowsybl networks."
            )

        self.base_net = base_net
        self.num_gens = len(base_net.poly_cost)
        self.permute_cols = self.base_net.poly_cost.columns[2:]

    def generate(
        self,
        example_generator: Generator[Union[pandapowerNet, Network], None, None],
    ) -> Generator[Union[pandapowerNet, Network], None, None]:
        """Generate a network with permuted generator cost coefficients.

        Args:
            example_generator: A generator producing example
                (load/topology/generation) scenarios to which generator
                cost coefficient permutations should be applied.

        Yields:
            An example scenario with cost coeffiecients in the
            poly_cost table permuted
        """
        for scenario in example_generator:
            new_idx = np.random.permutation(self.num_gens)
            scenario.poly_cost[self.permute_cols] = (
                scenario.poly_cost[self.permute_cols]
                .iloc[new_idx]
                .reset_index(drop=True)
            )
            yield scenario


class PerturbGenCostGenerator(GenerationGenerator):
    """Class for perturbing generator cost.

    This class is for generating different generation scenarios
    by randomly perturbing all the cost coeffiecient of generators
    in a power network by multiplying with a scaling factor sampled
    from a uniform distribution.
    """

    def __init__(self, base_net: Union[pandapowerNet, Network], sigma: float) -> None:
        """
        Initialize the gen-cost perturbation generator.

        Args:
            base_net: The base power network (pandapower or pypowsybl).
            sigma: Range parameter for uniform distribution sampling.

        Raises:
            NotImplementedError: If using pypowsybl network (poly_cost not available).
        """
        if isinstance(base_net, Network):
            raise NotImplementedError(
                "Generator cost perturbations are not supported for pypowsybl networks. "
                "PyPowSyBl does not have poly_cost tables and does not support OPF. "
                "Use 'none' as the generation_perturbation type for pypowsybl networks."
            )

        self.base_net = base_net
        self.num_gens = len(base_net.poly_cost)
        self.perturb_cols = self.base_net.poly_cost.columns[2:]
        self.lower = np.max([0.0, 1.0 - sigma])
        self.upper = 1.0 + sigma
        self.sample_size = [self.num_gens, len(self.perturb_cols)]

    def generate(
        self,
        example_generator: Generator[Union[pandapowerNet, Network], None, None],
    ) -> Generator[Union[pandapowerNet, Network], None, None]:
        """Generate a network with perturbed generator cost coefficients.

        Args:
            example_generator: A generator producing example
                (load/topology) scenarios to which generator cost coefficient
                perturbations should be added.

        Yields:
            An example scenario with cost coeffiecients in the poly_cost
            table perturbed by multiplying with a scaling factor.
        """
        for example in example_generator:
            scale_fact = np.random.uniform(
                low=self.lower,
                high=self.upper,
                size=self.sample_size,
            )
            example.poly_cost[self.perturb_cols] = (
                example.poly_cost[self.perturb_cols] * scale_fact
            )
            yield example
