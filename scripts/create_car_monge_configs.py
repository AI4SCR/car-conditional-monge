import os
from pathlib import Path

import yaml
from cmonge.utils import load_config


def change_and_save_configs(
    base_config_path,
    car_variants,
    controls,
    configs_save_path,
    adata_path,
    exp_base_dir,
):
    base_config = load_config(base_config_path)
    with open(car_variants) as f:
        cars = f.readlines()
    cars = [car.rstrip() for car in cars]

    for car in cars:
        config = base_config.copy()
        config.data.drug_condition = car
        if adata_path is not None:
            config.data.file_path = adata_path
        for control in controls:
            if car == control:
                continue
            exp_path = f"{exp_base_dir}{control}_{car}/"
            try:
                os.makedirs(exp_path)
            except FileExistsError:
                print("directory exists")
            config.data.control_condition = control
            config.model.checkpointing_path = f"{exp_path}/model/"
            config.logger_path = f"{exp_path}/logs.yaml"
            config_file = configs_save_path / f"{control}_{car}_config.yaml"

            # Save in configs folder
            with config_file.open("w") as f:
                yaml.dump(config.toDict(), f, default_flow_style=False)

            # Save in experiment folder
            with open(f"{exp_path}/config", "w") as f:
                yaml.dump(config.toDict(), f, default_flow_style=False)


if __name__ == "__main__":
    data_name = "CD8"
    setting = "Random"
    base_config_path = Path("cmonge/configs/monge-cars_{setting}.yml")
    configs_save_path = Path(
        "cmonge/configs/monge{setting}/{data_name}/",
    )

    exp_base_dir = Path("cmonge/monge/{setting}/{data_name}/")

    controls = ["NA-NA-NA"]
    adata_path = f"data/{data_name}.h5ad"
    car_variants = Path("cmonge/variants/CAR_variants.txt")

    for d in [configs_save_path, exp_base_dir]:
        try:
            os.makedirs(d)
        except FileExistsError:
            print("Directory exists, overwriting files")

    change_and_save_configs(
        base_config_path,
        car_variants,
        controls,
        configs_save_path,
        adata_path,
        exp_base_dir,
    )
