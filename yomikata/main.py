"""main.py
Main entry point for training
"""

import sys
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import mlflow
from datasets import load_dataset

from yomikata import utils
from yomikata.config import config, logger
from yomikata.dbert import dBert

# MLFlow model registry
mlflow.set_tracking_uri("file://" + str(config.RUN_REGISTRY.absolute()))


warnings.filterwarnings("ignore")


def train_model(
    model_name: "dBert",
    dataset_name: str = "",
    experiment_name: str = "baselines",
    run_name: str = "dbert-default",
    training_args: dict = {},
) -> None:
    """Train a model given arguments.

    Args:
        dataset_name (str): name of the dataset to be trained on. Defaults to the full dataset.
        args_fp (str): location of args.
        experiment_name (str): name of experiment.
        run_name (str): name of specific run in experiment.
    """

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
        artifacts_dir = Path(config.RUN_REGISTRY, experiment_id, run_id, "artifacts")

        # Initialize the model
        if model_name == "dBert":
            reader = dBert(reinitialize=True, artifacts_dir=artifacts_dir)
        else:
            raise ValueError("model_name must be dBert for now")

        # Load train val test data
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(Path(config.TRAIN_DATA_DIR, "train_" + dataset_name + ".csv")),
                "val": str(Path(config.VAL_DATA_DIR, "val_" + dataset_name + ".csv")),
                "test": str(Path(config.TEST_DATA_DIR, "test_" + dataset_name + ".csv")),
            },
        )

        # Train
        training_performance = reader.train(dataset, training_args=training_args)

        # general_performance = evaluate.evaluate(reader, max_evals=20)

        with tempfile.TemporaryDirectory() as dp:
            # reader.save(dp)
            # utils.save_dict(general_performance, Path(dp, "general_performance.json"))
            utils.save_dict(training_performance, Path(dp, "training_performance.json"))
            mlflow.log_artifacts(dp)


def get_artifacts_dir_from_run(run_id: str):
    """Load artifacts directory for a given run_id.

    Args:
        run_id (str): id of run to load artifacts from.

    Returns:
        Path: path to artifacts directory.

    """

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.RUN_REGISTRY, experiment_id, run_id, "artifacts")

    return artifacts_dir


if __name__ == "__main__":
    # get args filepath from input
    args_fp = sys.argv[1]

    # load the args_file
    args = Namespace(**utils.load_dict(filepath=args_fp)).__dict__

    # pop meta variables
    model_name = args.pop("model")
    dataset_name = args.pop("dataset")
    experiment_name = args.pop("experiment")
    run_name = args.pop("run")

    # Perform training
    train_model(
        model_name=model_name,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        run_name=run_name,
        training_args=args,
    )
