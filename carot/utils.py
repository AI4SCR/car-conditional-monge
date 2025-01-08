import random
from itertools import combinations

import numpy as np
import pandas as pd
from cmonge.metrics import average_r2, compute_scalar_mmd, wasserstein_distance


def get_source_target_transport(
    trainer,
    datamodule,
    conditions,
    n_samples=1,
    target=True,
    source=True,
    transport=True,
    max_sample: bool = False,
):
    if datamodule.data_config.split[2] > 0:
        print("Evaluating on test set")
        split_type = "test"
    elif datamodule.data_config.split[1] > 0:
        print("Evaluating on validation set")
        split_type = "valid"
    else:
        print("Evaluating on training set")
        split_type = "train"
    all_expr = []
    all_meta = []
    for i in range(n_samples):
        for condition in conditions:
            dm = datamodule.loaders[condition]
            cond_embeddings, num_contexts = trainer.embedding_module(condition)
            print(cond_embeddings.shape)
            batch_size = datamodule.data_config.batch_size
            if split_type == "valid":
                sel_target_cells = random.sample(
                    dm.target_valid_cells.tolist(),
                    batch_size,
                )
                sel_control_cells = random.sample(
                    dm.control_valid_cells.tolist(),
                    batch_size,
                )
            elif split_type == "test":
                sel_target_cells = random.sample(
                    dm.target_test_cells.tolist(), batch_size
                )
                sel_control_cells = random.sample(
                    dm.control_test_cells.tolist(), batch_size
                )
            elif split_type == "train":
                sel_target_cells = random.sample(
                    dm.target_train_cells.tolist(), batch_size
                )
                sel_control_cells = random.sample(
                    dm.control_train_cells.tolist(), batch_size
                )
            cond_expr = dm.adata[sel_target_cells].X
            cond_meta = dm.adata.obs.loc[sel_target_cells, :]
            source_expr = dm.adata[sel_control_cells].X
            source_meta = dm.adata.obs.loc[sel_control_cells, :]
            if target:
                cond_meta["sample_n"] = i
                cond_meta["dtype"] = "target"
                cond_meta["condition"] = condition
                all_expr.append(pd.DataFrame(cond_expr, columns=dm.adata.var_names))
                all_meta.append(cond_meta)
            if source:
                source_meta["dtype"] = "source"
                source_meta["sample_n"] = i
                source_meta["condition"] = condition
                all_meta.append(source_meta)
                all_expr.append(pd.DataFrame(source_expr, columns=dm.adata.var_names))
            if transport:
                trans = trainer.transport(source_expr, cond_embeddings, num_contexts=2)
                trans = datamodule.decoder(trans)
                trans_meta = cond_meta.copy()
                trans_meta["dtype"] = "transport"
                trans_meta["condition"] = condition
                trans_meta["sample_n"] = i
                all_expr.append(pd.DataFrame(trans, columns=dm.adata.var_names))
                all_meta.append(trans_meta)
        all_expr = pd.concat(all_expr).reset_index(drop=True)
        all_meta = pd.concat(all_meta).reset_index(drop=True)
        return all_expr, all_meta


def score_transports_and_targets_combinations(all_expr, all_meta):
    target_mmds = []
    target_wds = []
    target_r2s = []

    transport_mmds = []
    transport_wds = []
    transport_r2s = []

    tt_mmds = []
    tt_wds = []
    tt_r2s = []
    target_transport_cars = []

    for car1, car2 in list(combinations(all_meta["condition"].unique(), 2)):
        target1 = np.array(
            all_expr[(all_meta["condition"] == car1) & (all_meta["dtype"] == "target")]
        )
        target2 = np.array(
            all_expr[(all_meta["condition"] == car2) & (all_meta["dtype"] == "target")]
        )

        transport1 = np.array(
            all_expr[
                (all_meta["condition"] == car1) & (all_meta["dtype"] == "transport")
            ]
        )
        transport2 = np.array(
            all_expr[
                (all_meta["condition"] == car2) & (all_meta["dtype"] == "transport")
            ]
        )

        target_mmd = compute_scalar_mmd(target1, target2)
        target_wd = wasserstein_distance(target1, target2)
        target_r2 = average_r2(target1, target2)

        transport_mmd = compute_scalar_mmd(transport1, transport2)
        transport_wd = wasserstein_distance(transport1, transport2)
        transport_r2 = average_r2(transport1, transport2)

        target_mmds.append(target_mmd)
        target_wds.append(target_wd)
        target_r2s.append(target_r2)
        transport_mmds.append(transport_mmd)
        transport_wds.append(transport_wd)
        transport_r2s.append(transport_r2)

        if car1 not in target_transport_cars:
            tt_mmd = compute_scalar_mmd(target1, transport1)
            tt_wd = wasserstein_distance(target1, transport1)
            tt_r2 = average_r2(target1, transport1)

            tt_mmds.append(tt_mmd)
            tt_wds.append(tt_wd)
            tt_r2s.append(tt_r2)
            target_transport_cars.append(car1)

        if car2 not in target_transport_cars:
            tt_mmd = compute_scalar_mmd(target2, transport2)
            tt_wd = wasserstein_distance(target2, transport2)
            tt_r2 = average_r2(target2, transport2)

            tt_mmds.append(tt_mmd)
            tt_wds.append(tt_wd)
            tt_r2s.append(tt_r2)
            target_transport_cars.append(car2)

    scores = pd.DataFrame(
        {
            "scores": (
                target_mmds
                + transport_mmds
                + tt_mmds
                + target_r2s
                + transport_r2s
                + tt_r2s
                + target_wds
                + transport_wds
                + tt_wds
            ),
            "score": ["MMD"] * (2 * len(target_mmds) + len(tt_mmds))
            + ["R2"] * (2 * len(target_mmds) + len(tt_mmds))
            + ["WD"] * (2 * len(target_mmds) + len(tt_mmds)),
            "target_transport": (
                ["between targets"] * len(target_mmds)
                + ["between transports"] * len(transport_mmds)
                + ["between target transport"] * len(tt_mmds)
            )
            * 3,
        }
    )
    scores["scores"] = scores["scores"].astype(float)

    return scores
