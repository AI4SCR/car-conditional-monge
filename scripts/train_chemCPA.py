import pickle
import yaml
import typer
from chemCPA.experiments_run import ExperimentWrapper


def main(config_path):
    exp = ExperimentWrapper(init_all=False)

    with open(
        config_path,
        "r",
    ) as f:
        args = yaml.safe_load(f)


    exp.seed = 1337
    # loads the dataset splits
    exp.init_dataset(**args["dataset"])

    exp.init_drug_embedding(embedding=args["model"]["embedding"])
    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        enable_cpa_mode=args["model"]["enable_cpa_mode"],
        pretrained_model_path=args["model"]["pretrained_model_path"],
        pretrained_model_hashes=args["model"]["pretrained_model_hashes"],
    )
    # setup the torch DataLoader
    exp.update_datasets()

    results = exp.train(**args["training"])

    with open(f"{args['training']['save_dir']}/results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    typer.run(main)
