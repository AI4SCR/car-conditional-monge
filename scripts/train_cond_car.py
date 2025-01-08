import os
from pathlib import Path

import typer
from cmonge.utils import load_config
from loguru import logger

from carot.datasets.conditional_loader import ConditionalDataModule
from carot.trainers.conditional_monge_trainer import ConditionalMongeTrainer


def get_environ_var(env_var_name, fail_gracefully=True):
    try:
        assert (
            env_var_name in os.environ
        ), f"Environment variable ${env_var_name} not set, are you on a CCC job?"
        var = os.environ[env_var_name]
    except AssertionError:
        if not fail_gracefully:
            raise
        else:
            var = None

    return var


def train_ood_monge(config_path: Path):

    config = load_config(config_path)
    job_id = get_environ_var("LSB_JOBID", fail_gracefully=True)

    if len(config.logger_path) != 0:
        logger_path = Path(config.logger_path)
    else:
        # For submitting jobs
        data = config.data.file_path.split("/")[-1][:-5]
        embed = config.model.embedding.name
        logger_path = Path(f"cmonge/logs/cmonge/{data}_{embed}.yml")
    logger.info(f"Experiment: Training model on {config.data.drug_condition}")

    datamodule = ConditionalDataModule(config.data, config.condition, config.ae)
    trainer = ConditionalMongeTrainer(
        jobid=job_id,
        logger_path=logger_path,
        config=config.model,
        datamodule=datamodule,
    )
    trainer.train(datamodule)
    # Homogeneous evaluation
    trainer.evaluate(datamodule)
    trainer.save_checkpoint(config.model.checkpointing_args.checkpoint_dir)

    # Evaluate baselines
    # Model doesn't need to be trained, as there is no transport happening
    trainer.evaluate(datamodule, identity=True)
    car_conditions = config.condition.conditions
    for condition in car_conditions:
        print(condition)
        logger.info(f"Evaluating within condition {condition}")
        config.data.control_condition = condition
        config.condition.conditions = [condition]
        datamodule = ConditionalDataModule(config.data, config.condition, config.ae)
        trainer.evaluate(datamodule, identity=True)

    print("Training completed")


if __name__ == "__main__":
    typer.run(train_ood_monge)
