from dataclasses import asdict

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from flytekit import task, workflow

from california_housing_regression.pandera_workflows import (
    TARGET,
    Hyperparameters,
    Dataset,
    DatasetSplits,
    TrainingResult,
)


@task(cache=True, cache_version="1.0")
def get_dataset(test_size: float, random_state: int) -> DatasetSplits:
    dataset = fetch_california_housing(as_frame=True).frame
    return train_test_split(dataset, test_size=test_size, random_state=random_state)


@task(cache=True, cache_version="1.0")
def summarize_dataset(dataset: Dataset) -> pd.DataFrame:
    return dataset.describe()


@task(cache=True, cache_version="1.0")
def train_model(dataset: Dataset, hyperparameters: Hyperparameters) -> Ridge:
    model = Ridge(**asdict(hyperparameters))
    return model.fit(dataset.drop(TARGET, axis="columns"), dataset[TARGET])


@task(cache=True, cache_version="1.0")
def evaluate_model(dataset: Dataset, model: Ridge) -> float:
    features, target = dataset.drop(TARGET, axis="columns"), dataset[TARGET]
    # corrupt the features
    features = features.drop("Latitude", axis="columns")
    return mean_squared_error(target, model.predict(features))


@workflow
def main(
    hyperparameters: Hyperparameters,
    test_size: float = 0.2,
    random_state: int = 43,
)-> TrainingResult:
    train_dataset, test_dataset = get_dataset(
        test_size=test_size, random_state=random_state
    )

    summarize_dataset(dataset=train_dataset)
    summarize_dataset(dataset=test_dataset)

    model = train_model(dataset=train_dataset, hyperparameters=hyperparameters)
    train_mse = evaluate_model(dataset=train_dataset, model=model)
    test_mse = evaluate_model(dataset=test_dataset, model=model)

    return model, train_mse, test_mse


if __name__ == "__main__":
    print(f"trained model: {main(hyperparameters=Hyperparameters(alpha=10.0))}")
