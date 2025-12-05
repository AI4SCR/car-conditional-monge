import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import typer
import yaml
from chemCPA.data import (
    canonicalize_smiles,
    drug_names_to_once_canon_smiles,
    load_dataset_splits,
)
from chemCPA.embedding import get_chemical_representation
from chemCPA.model import ComPert
from chemCPA.paths import CHECKPOINT_DIR
from chemCPA.train import bool2idx, compute_prediction, repeat_n
from cmonge.metrics import average_r2, compute_scalar_mmd, wasserstein_distance
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(config):
    perturbation_key = config["dataset"]["data_params"]["perturbation_key"]
    dataset = sc.read(config["dataset"]["data_params"]["dataset_path"])
    key_dict = {
        "perturbation_key": perturbation_key,
    }
    return dataset, key_dict


def load_smiles(dataset, key_dict):
    perturbation_key = key_dict["perturbation_key"]

    # this is how the `canon_smiles_unique_sorted` is generated inside chemCPA.data.Dataset
    # we need to have the same ordering of SMILES, else the mapping to pathways will be off
    # when we load the Vanilla embedding. For the other embeddings it's not as important.
    drugs_names = np.array(dataset.obs[perturbation_key].values)
    drugs_names_unique = set()
    for d in drugs_names:
        [drugs_names_unique.add(i) for i in d.split("+")]
    drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))
    canon_smiles_unique_sorted = drugs_names_unique_sorted

    return canon_smiles_unique_sorted


def load_model(config, canon_smiles_unique_sorted, checkpoint_dir=CHECKPOINT_DIR):
    file_name = config["training"]["ckpt_name"]
    model_checkp = checkpoint_dir / (file_name + ".pt")

    embedding_model = config["model"]["embedding"]["model"]
    if embedding_model == "vanilla":
        embedding = None
    else:
        embedding = get_chemical_representation(
            smiles=canon_smiles_unique_sorted,
            embedding_model=config["model"]["embedding"]["model"],
            data_dir=config["model"]["embedding"]["directory"],
            device=device,
        )
    dumped_model = torch.load(model_checkp)
    if len(dumped_model) == 3:
        print("This model does not contain the covariate embeddings or adversaries.")
        state_dict, init_args, history = dumped_model
        cov_emb_available = False
    elif len(dumped_model) == 4:
        print("This model does not contain the covariate embeddings.")
        state_dict, cov_adv_state_dicts, init_args, history = dumped_model
        cov_emb_available = False
    elif len(dumped_model) == 5:
        (
            state_dict,
            cov_adv_state_dicts,
            cov_emb_state_dicts,
            init_args,
            history,
        ) = dumped_model
        cov_emb_available = True
        assert len(cov_emb_state_dicts) == 1
    append_layer_width = (
        config["dataset"]["n_vars"]
        if (config["model"]["append_ae_layer"] and config["model"]["load_pretrained"])
        else None
    )

    if embedding_model != "vanilla":
        state_dict.pop("drug_embeddings.weight")
    model = ComPert(
        **init_args,
        drug_embeddings=embedding,
        append_layer_width=append_layer_width,
        device=device,
    )
    model = model.eval()
    if cov_emb_available:
        for embedding_cov, state_dict_cov in zip(
            model.covariates_embeddings, cov_emb_state_dicts
        ):
            embedding_cov.load_state_dict(state_dict_cov)

    incomp_keys = model.load_state_dict(state_dict, strict=False)
    if embedding_model == "vanilla":
        assert (
            len(incomp_keys.unexpected_keys) == 0 and len(incomp_keys.missing_keys) == 0
        )
    else:
        # make sure we didn't accidentally load the embedding from the state_dict
        torch.testing.assert_allclose(model.drug_embeddings.weight, embedding.weight)
        assert (
            len(incomp_keys.missing_keys) == 1
            and "drug_embeddings.weight" in incomp_keys.missing_keys
        ), incomp_keys.missing_keys
        # assert len(incomp_keys.unexpected_keys) == 0, incomp_keys.unexpected_keys

    return model, embedding


# def compute_drug_embeddings(model, embedding, dosage=1e4):
#     all_drugs_idx = torch.tensor(list(range(len(embedding.weight))))
#     dosages = dosage * torch.ones((len(embedding.weight),))
#     # dosages = torch.ones((len(embedding.weight),))
#     with torch.no_grad():
#         # scaled the drug embeddings using the doser
#         transf_embeddings = model.compute_drug_embeddings_(
#             drugs_idx=all_drugs_idx, dosages=dosages
#         )
#         # apply drug embedder
#         # transf_embeddings = model.drug_embedding_encoder(transf_embeddings)
#     return transf_embeddings


def compute_pred(
    model,
    dataset,
    dosages=[1e4],
    cell_lines=None,
    genes_control=None,
    use_degs=True,
    verbose=True,
):
    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")

    cl_dict = {
        torch.Tensor([1, 0]): "D09",
        torch.Tensor([0, 1]): "D17",
    }

    if cell_lines is None:
        cell_lines = ["D09", "D17"]

    predictions_dict = {}
    drug_r2 = {}
    for cell_drug_dose_comb, category_count in tqdm(
        zip(*np.unique(dataset.pert_categories, return_counts=True))
    ):
        if dataset.perturbation_key is None:
            print("Perturbation key is none, breaking for loop")
            break

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            print("Category count <= 5, skipping")
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            print("Control condition --> skipped")
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        bool_de = dataset.var_names.isin(
            np.array(dataset.de_genes[cell_drug_dose_comb])
        )
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        if len(idx_de) < 2:
            print("idx_de < 2, skipping")
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]
        y_true = dataset.genes[idx_all, :].to(device=device)

        # cov_name = cell_drug_dose_comb.split("_")[0]
        # cond = dataset_ctrl.covariate_names["cell_type"] == cov_name
        # genes_control = dataset_ctrl.genes[cond]

        if genes_control is None:
            n_obs = y_true.size(0)
        else:
            assert isinstance(genes_control, torch.Tensor)
            n_obs = genes_control.size(0)

        emb_covs = [repeat_n(cov[idx], n_obs) for cov in dataset.covariates]

        if dataset.dosages[idx] not in dosages:
            print("Skipping dosage")
            continue

        stop = False
        for tensor, cl in cl_dict.items():
            if (tensor == dataset.covariates[0][idx]).all():
                if cl not in cell_lines:
                    stop = True
        if stop:
            continue

        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_obs).squeeze(),
                repeat_n(dataset.dosages[idx], n_obs).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_obs)

        # copies just the needed genes to GPU
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)

        if genes_control is None:
            # print("Predicting AE alike.")
            mean_pred, _ = compute_prediction(
                model,
                y_true,
                emb_drugs,
                emb_covs,
            )
        else:
            print("Predicting counterfactuals.")
            mean_pred, _ = compute_prediction(
                model,
                genes_control,
                emb_drugs,
                emb_covs,
            )

        mean_pred = mean_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        if use_degs:
            drug_r2[cell_drug_dose_comb] = average_r2(
                y_true[:, idx_de], mean_pred[:, idx_de]
            )
        else:
            drug_r2[cell_drug_dose_comb] = average_r2(y_true, mean_pred)

        predictions_dict[cell_drug_dose_comb] = [y_true, mean_pred, idx_de]
    return drug_r2, predictions_dict


def main(config_path, split):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set up
    dosages = [1]
    cell_lines = ["D09", "D17"]
    dataset, key_dict = load_dataset(config)
    config["dataset"]["n_vars"] = dataset.n_vars
    canon_smiles_unique_sorted = load_smiles(dataset, key_dict)

    # Load dataset
    data_params = config["dataset"]["data_params"]
    datasets = load_dataset_splits(**data_params, return_dataset=False)

    # Load checkpoint
    model, embedding = load_model(
        config,
        canon_smiles_unique_sorted,
        checkpoint_dir=Path(config["training"]["save_dir"]),
    )
    print(model)

    print(datasets)
    # Make predictions
    drug_r2, predictions_dict = compute_pred(
        model,
        datasets[split],
        genes_control=datasets["test_control"].genes,
        dosages=dosages,
        cell_lines=cell_lines,
        use_degs=True,
        verbose=True,
    )

    with open(
        f"{config['training']['save_dir']}/prediction_dict_{split}.pkl", "wb"
    ) as f:
        pickle.dump(predictions_dict, f)

    res = pd.DataFrame.from_dict(drug_r2, orient="index", columns=["carot_r2"])
    print(res.head())
    res[["car", "donor"]] = [c.split("_") for c in res.index]

    res.to_csv(f"{config['training']['save_dir']}/r2_mean_results_{split}.csv")


if __name__ == "__main__":

    # config_path = "/Users/alicedriessen/Projects/car-conditional-monge/configs/chemCPA_config.yml"
    # split = "test"

    typer.run(main)
