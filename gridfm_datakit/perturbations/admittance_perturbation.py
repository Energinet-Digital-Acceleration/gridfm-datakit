import numpy as np
import pandapower as pp
from pandapower.auxiliary import pandapowerNet
from pypowsybl.network import Network
from abc import ABC, abstractmethod
from typing import Generator, List, Union
from gridfm_datakit.network_interface import NetworkInterface


class AdmittanceGenerator(ABC):
    """Abstract base class for applying perturbations to line admittances."""

    def __init__(self) -> None:
        """Initialize the admittance generator."""
        pass

    @abstractmethod
    def generate(
        self,
        example_generator: Generator[Union[pandapowerNet, Network], None, None],
    ) -> Union[Generator[Union[pandapowerNet, Network], None, None], List[Union[pandapowerNet, Network]]]:
        """Generate admittance perturbations.

        Args:
            example_generator: A generator producing example (load/topology/generation)
            scenarios to which line admittance perturbations are added.

        Yields:
            An admittance-perturbed scenario.
        """
        pass


class NoAdmittancePerturbationGenerator(AdmittanceGenerator):
    """Generator that yields the original example generator without any perturbations."""

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
            (load/topology/generation) scenarios to which line admittance
            perturbations are added.

        Yields:
            The original example produced by the example_generator.
        """
        for example in example_generator:
            yield example


class PerturbAdmittanceGenerator(AdmittanceGenerator):
    """Class for applying perturbations to line admittances.

    This class is for generating different line admittance scenarios
    by applying perturbations to the resistance (R) and reactance (X)
    values of the lines.  Perturbations are applied as a scaling factor
    sampled from a uniform distribution with a given lower and upper
    bound.
    """

    def __init__(self, base_net: Union[pandapowerNet, Network], sigma: float) -> None:
        """
        Initialize the line admittance perturbation generator.

        Args:
            base_net: The base power network (pandapower or pypowsybl).
            sigma: Range parameter for uniform distribution sampling.
        """
        self.base_net = base_net
        adapter = NetworkInterface.create_adapter(base_net)
        lines = adapter.get_lines()

        # Determine column names based on network type
        # Pandapower: r_ohm_per_km, x_ohm_per_km (per unit length)
        # PyPowSyBl: r, x (absolute values in Ohms)
        if isinstance(base_net, pandapowerNet):
            self.r_col = "r_ohm_per_km"
            self.x_col = "x_ohm_per_km"
        else:  # pypowsybl
            self.r_col = "r"
            self.x_col = "x"

        self.r_original = lines[self.r_col].values
        self.x_original = lines[self.x_col].values
        self.lower = np.max([0.0, 1.0 - sigma])
        self.upper = 1.0 + sigma
        self.sample_size = self.r_original.shape[0]

    def generate(
        self,
        example_generator: Generator[Union[pandapowerNet, Network], None, None],
    ) -> Generator[Union[pandapowerNet, Network], None, None]:
        """Generate a network with perturbed line admittance values.

        Args:
            example_generator: A generator producing example
                (load/topology/generation) scenarios to which line admittance
                perturbations are added.

        Yields:
            An example scenario with random perturbations applied to line
            admittances.
        """
        for example in example_generator:
            adapter = NetworkInterface.create_adapter(example)
            lines = adapter.get_lines()

            # Apply perturbations to resistance and reactance
            lines[self.r_col] = np.random.uniform(
                self.lower * self.r_original,
                self.upper * self.r_original,
                self.r_original.shape[0],
            )

            lines[self.x_col] = np.random.uniform(
                self.lower * self.x_original,
                self.upper * self.x_original,
                self.x_original.shape[0],
            )

            # Update lines using adapter
            adapter.update_lines(lines)

            yield example
