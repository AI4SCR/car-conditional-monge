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
    grad_acc,
    layer_norm,
):
    base_config = load_config(base_config_path)
    with open(car_variants) as f:
        cars = f.readlines()
    cars = [car.rstrip() for car in cars]
    for car in cars:
        config = base_config.copy()
        config.ood_condition.split = [0.0, 1.0, 0.0]
        config.ood_condition.mode = "homogeneous"
        config.ood_condition.conditions = [car]
        config.condition.conditions = [c for c in cars if not c == car]
        config.condition.split = [0.8, 0.2, 0.0]
        config.data.file_path = adata_path
        for control in controls:
            if car == control:
                continue
            exp_path = exp_base_dir / f"{control}_{car}/"
            try:
                os.makedirs(exp_path)
            except FileExistsError:
                print("directory exists")
            config.data.control_condition = control
            config.model.checkpointing_args.checkpoint_dir = f"{exp_path}/model/"
            config.logger_path = f"{exp_path}/logs.yaml"
            config_file = Path(f"{configs_save_path}{car}_ood_config.yaml")
            config.model.mlp.layer_norm = layer_norm
            config.model.optim.grad_acc_steps = grad_acc
            config.model.num_train_iters = int(grad_acc) * int(
                config.model.num_train_iters
            )

            with config_file.open("w") as f:
                yaml.dump(config.toDict(), f, default_flow_style=False)

            with open(f"{exp_path}/config.yaml", "w") as f:
                yaml.dump(config.toDict(), f, default_flow_style=False)


if __name__ == "__main__":
    data_name = "CD8"
    setting = ""  # "sel_CARs_" or ""
    embedding = "16d"
    layer_norm = True
    grad_acc = 4
    lr_scheduler = "cosine"  # linear or cosine

    setting_name = f"{setting}LN_{layer_norm}_grad_acc_{grad_acc}_{lr_scheduler}"

    base_config_path = Path(
        "cmonge/configs/conditional-monge-cars-{data_name}_{setting}{embedding}.yml"
    )
    configs_save_path = Path(
        "cmonge/configs/cmonge_ood/{data_name}_{embedding}_{setting_name}_",
    )
    exp_base_dir = Path(
        "cmonge/cmonge_ood/{setting_name}/{data_name}_{embedding}_FuncScore/"
    )

    controls = ["NA-NA-NA"]
    adata_path = f"data/{data_name}.h5ad"
    car_variants = Path("cmonge/variants/CAR_variants.txt")

    try:
        os.makedirs(exp_base_dir)
    except FileExistsError:
        print("Directory exists, overwriting files")

    change_and_save_configs(
        base_config_path,
        controls,
        configs_save_path,
        adata_path,
        exp_base_dir,
        car_variants,
        grad_acc,
        layer_norm,
    )
