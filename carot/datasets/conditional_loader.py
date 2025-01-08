from typing import Dict

from cmonge.datasets.conditional_loader import (
    ConditionalDataModule as _ConditionalDataModule,
)

from .single_loader import DataModuleFactory


class ConditionalDataModule(_ConditionalDataModule):
    datamodule_factory: Dict[str, _ConditionalDataModule] = DataModuleFactory
