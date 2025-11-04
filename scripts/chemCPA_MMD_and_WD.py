from cmonge.evaluate import log_mean_metrics, log_metrics
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import pathlib
import pickle
from typing import Iterator
import typer
import yaml


def format_dict(dict_):
    dict_["wasserstein"] = [float(i) for i in dict_["wasserstein"]]
    dict_["mmd"] = [float(i) for i in dict_["mmd"]]
    dict_["sinkhorn_div"] = [float(i) for i in dict_["sinkhorn_div"]]
    dict_["monge_gap"] = [float(i) for i in dict_["monge_gap"]]
    dict_["drug_signature"] = [float(i) for i in dict_["drug_signature"]]
    dict_["r2"] = [float(i) for i in dict_["r2"]]
    dict_["mean_statistics"] = {
        k: float(v) for k, v in dict_["mean_statistics"].items()
    }


def sampler_iter(
    array: jnp.ndarray, batch_size: int, key: PRNGKeyArray
) -> Iterator[jnp.ndarray]:
    """Creates an inifinite dataloader with random sampling out of a jax array."""
    while True:
        k1, key = jax.random.split(key, 2)
        yield jax.random.choice(key=k1, a=array, shape=(batch_size,))


def evaluate_condition(
    loader_target: Iterator[jnp.ndarray],
    loader_transport: Iterator[jnp.ndarray],
    degs_idx,
    metrics: dict,
    n_samples=9,
):
    for enum, (target, transport) in enumerate(zip(loader_target, loader_transport)):

        target = target[:, degs_idx]
        transport = transport[:, degs_idx]

        log_metrics(metrics, target, transport)
        if enum > n_samples:
            break


def main(config: str, split: str):

    # donors = ["D09", "D17"]
    donors = ["D09"] # in case of dummy data
    metrics = {}
    key = jax.random.key(0)
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    with open(
        f"{config['training']['save_dir']}/prediction_dict_{split}.pkl", "rb"
    ) as f:
        predictions_dict = pickle.load(f)

    # All drug_dose combis in current OOD results
    # chemCPA condition also contains cell line, but we don't use it
    current_cars = list(
        set([c.split("_")[0] for c in predictions_dict.keys()])
    )

    # We gather all cells from same drug-dose combi as in CMonge
    for car in current_cars:
        drug_dose_conditions = [f"{car}_{donor}" for donor in donors]
        # Skip CTLA4-CTLA4-z for D17 because too few cells:
        # drug_dose_conditions = [dc for dc in drug_dose_conditions if not dc=='CTLA4-CTLA4-z_D17']
        all_target = []
        all_transport = []
        for cond in drug_dose_conditions:
            all_target.append(jnp.asarray(predictions_dict[cond][0]))
            all_transport.append(jnp.asarray(predictions_dict[cond][1]))
        all_target = jnp.vstack(all_target)
        all_transport = jnp.vstack(all_transport)
        degs_idx = predictions_dict[cond][2]

        # Evaluate drug_dose condition (without cell line split)
        metrics[car] = {}
        metrics[car]["car"] = car
        metrics[car]["wasserstein"] = []
        metrics[car]["mmd"] = []
        metrics[car]["sinkhorn_div"] = []
        metrics[car]["monge_gap"] = []
        metrics[car]["drug_signature"] = []
        metrics[car]["r2"] = []
        metrics[car]["mean_statistics"] = {}

        key, k1, k2 = jax.random.split(key, 3)
        target_loader = sampler_iter(all_target, batch_size=512, key=k1)
        transport_loader = sampler_iter(all_transport, batch_size=512, key=k2)

        evaluate_condition(
            target_loader, transport_loader, degs_idx, metrics[car]
        )
        log_mean_metrics(metrics[car])

        format_dict(metrics[car])

    with open(f"{config['training']['save_dir']}/cmonge_eval_{split}.yml", "w") as f:
        yaml.safe_dump(metrics, f, default_flow_style=False)
    print("Metrics saved")


if __name__ == "__main__":

    typer.run(main)
