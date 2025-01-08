import os
from pathlib import Path

import yaml
from cmonge.utils import load_config


def change_and_save_configs(
    base_config_path,
    controls,
    configs_save_path,
    adata_path,
    exp_base_dir,
    car_variants,
    setting,
    layer_norm,
    grad_acc,
    lr_scheduler,
):
    base_config = load_config(base_config_path)
    config = base_config.copy()

    if "sel_cars" not in setting.lower():

        with open(car_variants) as f:
            cars = f.readlines()
        cars = [car.rstrip() for car in cars]
        config.condition.conditions = cars

    config.condition.split = [0.8, 0.2, 0.0]
    config.data.file_path = adata_path
    config.model.lr_scheduler.name = lr_scheduler
    if lr_scheduler.lower() == "linear":
        config.model.lr_scheduler.name
        config.model.lr_scheduler.kwargs.pct_start = 0.1
        config.model.lr_scheduler.kwargs.pct_final = 0.8
    for control in controls:
        exp_path = exp_base_dir
        # try:
        #     os.makedirs(exp_path)
        # except FileExistsError:
        #     print("directory exists")
        config.data.control_condition = control
        config.model.checkpointing_args.checkpoint_dir = f"{exp_path}/model/"
        config.logger_path = f"{exp_path}/logs.yaml"
        config.model.mlp.layer_norm = layer_norm
        config.model.optim.grad_acc_steps = grad_acc
        config.model.num_train_iters = int(grad_acc) * int(config.model.num_train_iters)

        config_file = Path(f"{configs_save_path}config.yaml")
        with config_file.open("w") as f:
            yaml.dump(config.toDict(), f, default_flow_style=False)

        with open(f"{exp_path}/config.yaml", "w") as f:
            yaml.dump(config.toDict(), f, default_flow_style=False)


if __name__ == "__main__":
    data_name = "CD8"
    setting = "sel_CARs_"  # "sel_CARs_" or ""
    embedding = "esm_small_tail_dim"
    layer_norm = True
    grad_acc = 4
    lr_scheduler = "cosine"  # linear or cosine

    setting_name = f"{setting}LN_{layer_norm}_grad_acc_{grad_acc}_{lr_scheduler}"

    base_config_path = Path(
        "cmonge/configs/conditional-monge-cars-{data_name}_{setting}{embedding}.yml"
    )
    configs_save_path = Path(
        "cmonge/configs/cmonge/{data_name}_{embedding}_{setting_name}_",
    )
    exp_base_dir = Path(
        "cmonge/cmonge/{setting_name}/{data_name}_{embedding}_FuncScore/"
    )

    controls = ["NA-NA-NA"]
    adata_path = f"data/{data_name}.h5ad"
    car_variants = Path("cmonge/variants/CAR_variants.txt")

    for d in [exp_base_dir]:
        try:
            os.makedirs(d)
        except FileExistsError:
            print("Directory exists")

    change_and_save_configs(
        base_config_path,
        controls,
        configs_save_path,
        adata_path,
        exp_base_dir,
        car_variants,
        setting,
        layer_norm,
        grad_acc,
        lr_scheduler,
    )
