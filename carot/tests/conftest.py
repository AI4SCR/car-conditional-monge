from pathlib import Path

import pytest
from cmonge.utils import load_config

# from carot.datasets.conditional_loader import ConditionalDataModule
# from carot.datasets.single_loader import CarModule


@pytest.fixture
def synthetic_config():
    config_path = Path("carot/tests/configs/synthetic.yml")
    config = load_config(config_path)
    return config


@pytest.fixture
def cond_synthetic_config():
    config_path = Path("carot/tests/configs/conditional_synthetic.yml")
    config = load_config(config_path)
    return config


# @pytest.fixture
# def synthetic_data(synthetic_config):
#     module = CarModule(synthetic_config.data)
#     return module


# @pytest.fixture
# def cond_synthetic_data(cond_synthetic_config):
#     module = ConditionalDataModule(
#         cond_synthetic_config.data, cond_synthetic_config.condition
#     )
#     return module
