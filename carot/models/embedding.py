from pathlib import Path

import jax.numpy as jnp
import pandas as pd
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.models.embedding import BaseEmbedding
from loguru import logger


class CAR11DimEmbedding(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            labels = datamodule.train_conditions

            car_11d = pd.DataFrame([self.encode_car_11dim(label) for label in labels]).T
            car_11d.colums = labels
            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / name
            car_11d.to_csv(model_dir)
        else:
            model_dir = self.model_dir / name
            car_11d = pd.read_csv(model_dir)
            car_11d = car_11d.drop(columns=["Unnamed: 0"])

        for index, row in car_11d.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def encode_car_11dim(self, car: str) -> list:
        """
        Compute one-hot encoding of CAR variant on 15 bits.
        Use alphabetical order of CAR domains: 41BB, CD28, CD40, CTLA4, IL15RA.
        For each domain there are 3 bits:
            - Domain present
            - 1st position
            - 2nd position
        So the three bits of CAR with domain A in the first position would be [1,1,0].
        A in second position would be [1,0,1]
        and for the CAR with both domains A [1,1,1].
        The 3 bits for the cars are concatenated into 15 bit. Then the l6th bit is to
        indicate wether CD3z (`z`) is present.
        0 everywhere is TCR-
        """
        all_domains = ["41BB", "CD28", "CD40", "CTLA4", "IL15RA"]
        car_variant = car.split("-")

        encoding = [0] * 11

        if car_variant[0] != "NA":
            # First mark first domain
            index_1 = all_domains.index(car_variant[0])
            encoding[index_1] = 1

        # Mark second domain if present
        if car_variant[1] != "NA":
            index_2 = all_domains.index(car_variant[1])
            encoding[index_2 + 5] = 1

        if car_variant[2] == "z":
            encoding[-1] = 1

        return encoding

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch, 1


class CAR16DimEmbedding(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            labels = datamodule.train_conditions

            car_16d = pd.DataFrame([self.encode_car_16dim(label) for label in labels]).T
            car_16d.colums = labels
            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / name
            car_16d.to_csv(model_dir)
        else:
            model_dir = self.model_dir / name
            car_16d = pd.read_csv(model_dir)
            car_16d = car_16d.drop(columns=["Unnamed: 0"])

        for index, row in car_16d.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def encode_car_16dim(self, car):
        """
        Compute one-hot encoding of CAR variant on 16 bits.
        Use alphabetical order of CAR domains: 41BB, CD28, CD40, CTLA4, IL15RA.
        For each domain there are 3 bits:
            - Domain present
            - 1st position
            - 2nd position
        So the three bits of CAR with domain A in the first position would be [1,1,0].
        A in second position would be [1,0,1]
        and for the CAR with both domains A [1,1,1].
        The 3 bits for the cars are concatenated into 15 bit. Then the 16th bit is to
        indicate wether CD3z (`z`) is present.
        0 everywhere is TCR-
        """
        all_domains = ["41BB", "CD28", "CD40", "CTLA4", "IL15RA"]
        car_variant = car.split("-")

        encoding = [0] * 16

        if car_variant[0] != "NA":
            # First mark first domain
            index_1 = all_domains.index(car_variant[0])
            encoding[index_1 * 3] = 1
            encoding[index_1 * 3 + 1] = 1

        # Mark second domain if present
        if car_variant[1] != "NA":
            index_2 = all_domains.index(car_variant[1])
            encoding[index_2 * 3] = 1
            encoding[index_2 * 3 + 2] = 1

        if car_variant[2] == "z":
            encoding[-1] = 1

        return encoding

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch, 1


class CarEsmSmall(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            logger.error(
                """ESM embedding only works with checkpoint,
                please save a pre-computed embedding"""
            )
        else:
            model_dir = self.model_dir / name
            embed = pd.read_csv(model_dir)
            embed = embed.drop(columns=["Unnamed: 0"])

        for index, row in embed.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch, 1


class MetaDataEmbedding(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        dataset = datamodule.data_config.file_path.split("/")[-1][:-5]
        if not checkpoint:
            adata = datamodule.data_config.parent
            group = "CAR_Variant"
            cont_scores = [
                "Cytotoxicity_1",
                "Proinflamatory_2",
                "Memory_3",
                "CD4_Th1_4",
                "CD4_Th2_5",
                "S.Score",
                "G2M.Score",
            ]
            fraction_scores = ["Donor", "Time", "Phase", "ident", "subset"]

            means = adata.obs[[group] + cont_scores].groupby(group).mean()
            stds = adata.obs[[group] + cont_scores].groupby(group).std()
            cont_features = means.merge(
                stds, left_index=True, right_index=True, suffixes=("_mean", "_std")
            )

            group_size = adata.obs.groupby(group).size()
            all_cat_counts = []
            for cat in fraction_scores:
                temp = (
                    adata.obs.groupby([group, cat], observed=False)
                    .size()
                    .reset_index(drop=False)
                )
                temp = pd.pivot_table(
                    data=temp, index=group, columns=cat, values=0, observed=False
                )
                all_cat_counts.append(temp)

            cat_features = pd.concat(all_cat_counts, axis=1)
            cat_features = cat_features.div(group_size, axis=0)

            embedding = cont_features.merge(
                cat_features, left_index=True, right_index=True
            ).T

            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / f"{dataset}_{name}"
            embedding.to_csv(model_dir)
        else:
            model_dir = self.model_dir / f"{dataset}_{name}"
            embedding = pd.read_csv(model_dir)
            embedding = embedding.drop(columns=["Unnamed: 0"])

        for index, row in embedding.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch, 1


EmbeddingFactory = {
    "embed_11d": CAR11DimEmbedding,
    "embed_16d": CAR16DimEmbedding,
    "esm_small": CarEsmSmall,
    "esm_small_full_dim": CarEsmSmall,
    "esm_small_full_seq": CarEsmSmall,
    "esm_small_tail_dim": CarEsmSmall,
    "esm_small_tail_seq": CarEsmSmall,
    "esm2_t33_650M_UR50D_tail_dim": CarEsmSmall,
    "esm2_t48_15B_UR50D_tail_dim": CarEsmSmall,
    "metadata": MetaDataEmbedding,
}
