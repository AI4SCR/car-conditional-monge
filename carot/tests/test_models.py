# import os
# from pathlib import Path

# from cmonge.trainers.ot_trainer import MongeMapTrainer

# from carot.trainers.conditional_monge_trainer import ConditionalMongeTrainer


# def test_conditional_model_training(cond_synthetic_config, cond_synthetic_data):

#     logger_path = Path(cond_synthetic_config.logger_path)

#     trainer = ConditionalMongeTrainer(
#         jobid=1,
#         logger_path=logger_path,
#         config=cond_synthetic_config.model,
#         datamodule=cond_synthetic_data,
#     )

#     trainer.train(cond_synthetic_data)
#     trainer.evaluate(cond_synthetic_data)

#     os.remove(logger_path)


# def test_model_training(synthetic_config, synthetic_data):

#     logger_path = Path(synthetic_config.logger_path)

#     trainer = MongeMapTrainer(
#         jobid=1, logger_path=logger_path, config=synthetic_config.model
#     )

#     trainer.train(synthetic_data)
#     trainer.evaluate(synthetic_data)

#     os.remove(logger_path)
