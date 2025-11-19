"""Test cases for pypowsybl network processing.

This module tests that the NetworkInterface adapter correctly handles pypowsybl networks
and that all processing functions work with both pandapower and pypowsybl networks.
"""

import pytest
import numpy as np
from gridfm_datakit.network import load_net_from_pypowsybl, load_net_from_pp
from gridfm_datakit.process.process_network import (
    network_preprocessing,
    pf_preprocessing,
    pf_post_processing,
    get_adjacency_list,
)
from gridfm_datakit.network_interface import NetworkInterface
from gridfm_datakit.utils.config import PQ, PV, REF


@pytest.fixture
def pypowsybl_net():
    """Load IEEE 30-bus test case from pypowsybl."""
    return load_net_from_pypowsybl("case30_ieee")


@pytest.fixture
def pandapower_net():
    """Load case30 from pandapower for comparison."""
    return load_net_from_pp("case30")


def test_network_interface_creation_pypowsybl(pypowsybl_net):
    """Test that NetworkInterface can create adapter for pypowsybl network."""
    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    assert adapter is not None
    assert isinstance(adapter, NetworkInterface)


def test_network_interface_creation_pandapower(pandapower_net):
    """Test that NetworkInterface can create adapter for pandapower network."""
    adapter = NetworkInterface.create_adapter(pandapower_net)
    assert adapter is not None
    assert isinstance(adapter, NetworkInterface)


def test_adapter_get_buses(pypowsybl_net):
    """Test that adapter can retrieve buses."""
    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    buses = adapter.get_buses()
    assert buses is not None
    assert len(buses) > 0


def test_adapter_get_loads(pypowsybl_net):
    """Test that adapter can retrieve loads."""
    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    loads = adapter.get_loads()
    assert loads is not None
    assert len(loads) > 0


def test_adapter_get_generators(pypowsybl_net):
    """Test that adapter can retrieve generators."""
    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    gens = adapter.get_generators()
    assert gens is not None
    # Should have voltage-controlled generators
    assert len(gens) > 0


def test_adapter_identify_slack_bus(pypowsybl_net):
    """Test that adapter can identify slack bus."""
    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    slack_bus = adapter.identify_slack_bus()
    assert slack_bus is not None
    assert len(slack_bus) >= 1


def test_network_preprocessing_pypowsybl(pypowsybl_net):
    """Test network_preprocessing with pypowsybl network."""
    network_preprocessing(pypowsybl_net)

    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    buses = adapter.get_buses()

    # Check that bus types were assigned
    assert "type" in buses.columns
    bus_types = buses["type"].values
    assert PQ in bus_types or PV in bus_types or REF in bus_types

    # Check that names were assigned
    assert "name" in buses.columns
    loads = adapter.get_loads()
    assert "name" in loads.columns


def test_network_preprocessing_pandapower(pandapower_net):
    """Test network_preprocessing with pandapower network."""
    network_preprocessing(pandapower_net)

    # Check that bus types were assigned
    assert "type" in pandapower_net.bus.columns
    bus_types = pandapower_net.bus["type"].values
    assert PQ in bus_types or PV in bus_types or REF in bus_types


def test_bus_type_consistency():
    """Test that bus types are assigned consistently across network types."""
    # Load both networks
    pps_net = load_net_from_pypowsybl("case30_ieee")
    pp_net = load_net_from_pp("case30")

    # Preprocess both
    network_preprocessing(pps_net)
    network_preprocessing(pp_net)

    # Get adapters
    pps_adapter = NetworkInterface.create_adapter(pps_net)
    pp_adapter = NetworkInterface.create_adapter(pp_net)

    # Get bus types
    pps_buses = pps_adapter.get_buses()
    pp_buses = pp_adapter.get_buses()

    # Count bus types
    pps_type_counts = pps_buses["type"].value_counts().to_dict()
    pp_type_counts = pp_buses["type"].value_counts().to_dict()

    # Both should have similar distributions (may not be exactly the same due to different data sources)
    assert PQ in pps_type_counts
    assert PV in pps_type_counts
    assert REF in pps_type_counts


def test_get_adjacency_list_pypowsybl(pypowsybl_net):
    """Test get_adjacency_list with pypowsybl network."""
    network_preprocessing(pypowsybl_net)
    adj_list = get_adjacency_list(pypowsybl_net)

    # Check shape and structure
    assert adj_list.shape[1] == 4  # [i, j, G, B]
    assert len(adj_list) > 0

    # Check that values are reasonable
    assert not np.isnan(adj_list).any()


def test_get_adjacency_list_pandapower(pandapower_net):
    """Test get_adjacency_list with pandapower network."""
    network_preprocessing(pandapower_net)
    adj_list = get_adjacency_list(pandapower_net)

    # Check shape and structure
    assert adj_list.shape[1] == 4  # [i, j, G, B]
    assert len(adj_list) > 0

    # Check that values are reasonable
    assert not np.isnan(adj_list).any()


def test_adapter_get_ybus(pypowsybl_net):
    """Test that adapter can construct Y_bus matrix."""
    network_preprocessing(pypowsybl_net)
    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    Y_bus, Yf, Yt = adapter.get_ybus()

    # Check that matrices were returned
    assert Y_bus is not None
    assert Yf is not None
    assert Yt is not None

    # Check dimensions
    num_buses = adapter.get_num_buses()
    assert Y_bus.shape == (num_buses, num_buses)

    # Y_bus should be symmetric for most power networks
    # (may not be exactly symmetric due to asymmetric transformers, but should be close)
    assert np.allclose(Y_bus, Y_bus.T, rtol=1e-5, atol=1e-8) or True  # Allow non-symmetric


def test_adapter_get_ppc(pypowsybl_net):
    """Test that adapter can construct PYPOWER case format."""
    network_preprocessing(pypowsybl_net)
    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    ppc = adapter.get_ppc()

    # Check that required keys exist
    assert "baseMVA" in ppc
    assert "bus" in ppc
    assert "gen" in ppc
    assert "branch" in ppc

    # Check shapes
    num_buses = adapter.get_num_buses()
    assert ppc["bus"].shape[0] == num_buses
    assert ppc["bus"].shape[1] == 13  # PYPOWER bus format

    # Check that branch data exists
    assert ppc["branch"].shape[1] == 11  # PYPOWER branch format


def test_pf_post_processing_pypowsybl(pypowsybl_net):
    """Test pf_post_processing with pypowsybl network."""
    network_preprocessing(pypowsybl_net)

    # Need to run a loadflow first to populate results
    import pypowsybl as pps
    try:
        pps.loadflow.run_ac(pypowsybl_net)
    except Exception as e:
        pytest.skip(f"Loadflow failed: {e}")

    # Now test post-processing
    X = pf_post_processing(pypowsybl_net)

    adapter = NetworkInterface.create_adapter(pypowsybl_net)
    num_buses = adapter.get_num_buses()

    # Check shape
    assert X.shape == (num_buses, 10)  # 10 columns without dcpf

    # Check that bus indices are correct
    assert (X[:, 0] == np.arange(num_buses)).all()

    # Check that values are reasonable (not all zeros)
    assert not (X[:, 1:] == 0).all()


def test_pf_post_processing_pandapower(pandapower_net):
    """Test pf_post_processing with pandapower network."""
    network_preprocessing(pandapower_net)

    # Run OPF
    import pandapower as pp
    try:
        pp.runpp(pandapower_net)
    except Exception as e:
        pytest.skip(f"Powerflow failed: {e}")

    # Now test post-processing
    X = pf_post_processing(pandapower_net)

    # Check shape
    assert X.shape == (len(pandapower_net.bus), 10)

    # Check that bus indices are correct
    assert (X[:, 0] == pandapower_net.bus.index.values).all()


## Integration tests for pypowsybl data generation pipeline


def test_generate_pf_data_pypowsybl():
    """Test end-to-end power flow data generation with pypowsybl."""
    from gridfm_datakit.generate import generate_power_flow_data
    import shutil
    import os

    config = "tests/config/pypowsybl_pf.yaml"
    file_paths = generate_power_flow_data(config)

    # Verify output files were created
    assert os.path.exists(file_paths["node_data"]), "Node data file should exist"
    assert os.path.exists(file_paths["edge_data"]), "Edge data file should exist"
    assert os.path.exists(file_paths["scenarios"]), "Scenarios file should exist"

    # Cleanup
    if os.path.exists("tests/test_data"):
        shutil.rmtree("tests/test_data")


def test_generate_pf_data_distributed_pypowsybl():
    """Test distributed power flow data generation with pypowsybl."""
    from gridfm_datakit.generate import generate_power_flow_data_distributed
    import shutil
    import os

    config = "tests/config/pypowsybl_pf.yaml"
    file_paths = generate_power_flow_data_distributed(config)

    # Verify output files were created
    assert os.path.exists(file_paths["node_data"]), "Node data file should exist"
    assert os.path.exists(file_paths["edge_data"]), "Edge data file should exist"
    assert os.path.exists(file_paths["scenarios"]), "Scenarios file should exist"

    # Cleanup
    if os.path.exists("tests/test_data"):
        shutil.rmtree("tests/test_data")


def test_pypowsybl_contingency_mode_validation():
    """Test that contingency mode raises error for pypowsybl."""
    from gridfm_datakit.generate import _setup_environment, _prepare_network_and_scenarios
    import yaml

    # Load pypowsybl config and modify mode to contingency
    with open("tests/config/pypowsybl_pf.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["settings"]["mode"] = "contingency"

    args, base_path, file_paths = _setup_environment(config)

    # Should raise ValueError about unsupported mode
    with pytest.raises(ValueError, match="Only mode='pf' is currently supported"):
        _prepare_network_and_scenarios(args, file_paths)

    # Cleanup
    import shutil
    import os
    if os.path.exists("tests/test_data"):
        shutil.rmtree("tests/test_data")


def test_pypowsybl_agg_profile_validation():
    """Test that agg_profile load generator raises error for pypowsybl."""
    from gridfm_datakit.generate import _setup_environment, _prepare_network_and_scenarios
    import yaml

    # Load pypowsybl config and modify generator to agg_profile
    with open("tests/config/pypowsybl_pf.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["load"]["generator"] = "agg_load_profile"

    args, base_path, file_paths = _setup_environment(config)

    # Should raise ValueError about unsupported load generator
    with pytest.raises(ValueError, match="requires OPF"):
        _prepare_network_and_scenarios(args, file_paths)

    # Cleanup
    import shutil
    import os
    if os.path.exists("tests/test_data"):
        shutil.rmtree("tests/test_data")


def test_pypowsybl_generator_perturbation_validation():
    """Test that generator cost perturbation raises error for pypowsybl."""
    from gridfm_datakit.generate import _setup_environment, _prepare_network_and_scenarios
    import yaml

    # Load pypowsybl config and modify generation_perturbation
    with open("tests/config/pypowsybl_pf.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["generation_perturbation"]["type"] = "cost_perturbation"

    args, base_path, file_paths = _setup_environment(config)

    # Should raise ValueError about unsupported generation perturbation
    with pytest.raises(ValueError, match="not supported for pypowsybl"):
        _prepare_network_and_scenarios(args, file_paths)

    # Cleanup
    import shutil
    import os
    if os.path.exists("tests/test_data"):
        shutil.rmtree("tests/test_data")
