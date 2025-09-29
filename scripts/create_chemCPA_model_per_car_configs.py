import os
from pathlib import Path

import yaml
from cmonge.utils import load_config


def change_and_save_configs(
    base_config_path,
    car_variants,
    configs_save_path,
    adata_path,
    exp_base_dir,
):
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    with open(car_variants) as f:
        cars = f.readlines()
    cars = [car.rstrip() for car in cars]

    for car in cars:
        config = base_config.copy()        
        config["dataset"]["data_params"]["dataset_path"]= adata_path
        config["dataset"]["data_params"]["split_key"] = f"{car}_ID"
    
        exp_path = f"{exp_base_dir}/{car}/"
        try:
            os.makedirs(exp_path)
        except FileExistsError:
            print("directory exists")
        config["training"]["save_dir"] = exp_path
        config["model"]["pretrained_model_path"] = exp_path
        config_file = configs_save_path / f"{car}_config.yml"

        # Save in configs folder
        with config_file.open("w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save in experiment folder
        with open(f"{exp_path}/config.yml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    data_name = "CD4" # CD4, CD8
    base_config_path = Path(f"/Users/alicedriessen/Projects/car-conditional-monge/configs/chemCPA_{data_name}_ID_config_dummy_donor.yml")
    configs_save_path = Path(
        f"/Users/alicedriessen/Projects/car-conditional-monge/configs/chemCPA_per_car/{data_name}/",
    )

    exp_base_dir = Path(f"/Users/alicedriessen/Box/LegacyFromOldColleagues/Alice/CAR_Tcells/Model/chemCPA/model_per_car/{data_name}")
    adata_path = f"/Users/alicedriessen/Box/LegacyFromOldColleagues/Alice/CAR_Tcells/Model/OT/{data_name}_chemCPA_anno.h5ad"
    car_variants = Path("/Users/alicedriessen/Box/LegacyFromOldColleagues/Alice/CAR_Tcells/Model/OT/CAR_variants.txt")

    for d in [configs_save_path, exp_base_dir]:
        try:
            os.makedirs(d)
        except FileExistsError:
            print("Directory exists, overwriting files")

    change_and_save_configs(
        base_config_path,
        car_variants,
        configs_save_path,
        adata_path,
        exp_base_dir,
    )
