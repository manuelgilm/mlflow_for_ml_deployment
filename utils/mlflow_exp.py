from typing import Dict
from typing import Optional
from mlflow.entities import Experiment
import mlflow


def get_or_create_experiment(name: str, tags: Optional[Dict[str, str]]) -> Experiment:
    """
    Get or create an MLflow experiment with the given name and tags.
    If the experiment already exists, it will be returned. Otherwise,
    a new experiment will be created with the specified name and tags.

    :param name: The name of the experiment.
    :param tags: A dictionary of tags to associate with the experiment.
    :return: The MLflow experiment object.
    """

    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        # Create a new experiment with the specified name and tags
        experiment_id = mlflow.create_experiment(name=name, tags=tags)
        print(f"Experiment '{name}' created with ID: {experiment_id}")

    experiment = mlflow.set_experiment(experiment_name=name)
    return experiment
