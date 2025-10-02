from pathlib import Path
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import scanpy as sc
from anndata import AnnData
from cmonge.datasets.single_loader import AbstractDataModule
from cmonge.trainers.ae_trainer import AETrainerModule
from cmonge.utils import load_config
from dotmap import DotMap


class CarModule(AbstractDataModule):
    def __init__(self, config: DotMap) -> None:
        super().__init__()
        self.setup(**config)

    def setup(
        self,
        name: str,
        file_path: Path,
        drugs_path: Path,
        features: Path,
        split: list[float],
        batch_size: int,
        drug_col: str,
        drug_condition: str,
        control_condition: str,
        ae: bool,
        seed: int,
        ae_config_path: Optional[Path] = None,
        reduction: Optional[str] = None,
        parent: Optional[AnnData] = None,
        parent_reducer: Optional[str] = None,
    ) -> None:
        self.name = name
        self.file_path = file_path
        self.split = split
        self.batch_size = batch_size
        self.features_path = features
        self.drugs_path = drugs_path
        self.drug_col = drug_col
        self.drug_condition = drug_condition
        self.control_condition = control_condition
        self.ae = ae
        self.ae_config = load_config(ae_config_path) if ae_config_path else None
        self.reduction = reduction
        self.parent = parent
        self.parent_reducer = parent_reducer
        self.seed = seed
        self.key = jax.random.PRNGKey(self.seed)

        self.loader()
        self.preprocesser()
        self.splitter()
        self.reducer()

    def loader(self) -> None:
        if self.parent:
            self.adata = self.parent
        else:
            self.adata = sc.read_h5ad(self.file_path)
        with open(self.features_path) as f:
            features = f.readlines()
        self.features = [feature.rstrip() for feature in features]

        with open(self.drugs_path) as f:
            drugs = f.readlines()
        self.drugs = [drug.rstrip() for drug in drugs]

    def preprocesser(self) -> None:
        self.adata = self.adata[:, self.features].copy()
        self.adata.X = self.adata.layers["logcounts"]
        self.adata.X = jnp.asarray(self.adata.X.todense())

    def reducer(self):
        """Sets up dimensionality reduction, either with PCA, AE or identity."""
        if self.reduction == "pca":
            self.pca_means = self.adata.X.mean(axis=0)
            self.encoder = lambda x: (x - self.pca_means) @ self.adata.varm["PCs"]
            self.decoder = lambda x: x @ self.adata.varm["PCs"].T + self.pca_means
        elif self.reduction == "ae":
            if self.parent_reducer:
                trainer = self.parent_reducer
            else:
                trainer = AETrainerModule(self.ae_config)
                trainer.load_model(self.name, self.drug_condition)
            model = trainer.model.bind({"params": trainer.state.params})
            self.encoder = lambda x: model.encoder(x)
            self.decoder = lambda x: model.decoder(x)
        else:
            self.encoder = lambda x: x
            self.decoder = lambda x: x

    def train_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        train_loaders = self.get_loaders_by_type("train")
        return train_loaders

    def valid_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        valid_loaders = self.get_loaders_by_type("valid")
        return valid_loaders

    def test_dataloaders(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        test_loaders = self.get_loaders_by_type("test", batch_size)
        return test_loaders


DataModuleFactory = {
    "car": CarModule,
}
