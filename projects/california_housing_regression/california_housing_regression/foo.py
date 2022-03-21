from flytekit import task, workflow


@task
def get_data(): ...

@task
def process_data(): ...

@task
def train_model(): ...

@workflow
def training_workflow():
    data = get_data()
    processed_data = process_data(data=data)
    model = train_model(processed_data=processed_data)


import logging
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
from typing import Annotated, NamedTuple

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from flytekit import task, workflow, kwtypes


logger = logging.getLogger("flytekit")
logger.setLevel(logging.ERROR)


@dataclass_json
@dataclass
class Hyperparameters:
    alpha: float
    random_state: int = 42


Dataset = Annotated[
    pd.DataFrame,
    kwtypes(
        Latitude=float,
        Longitude=float,
        AveBedrms=float,
        AveOccup=float,
        AveRooms=float,
        HouseAge=float,
        MedInc=float,
        MedHouseVal=float,
    )
]

TARGET = "MedHouseVal"

DatasetSplits = NamedTuple("DatasetSplits", train=Dataset, test=Dataset)
TrainingResult = NamedTuple("TrainingResult", model=Ridge, train_mse=float, test_mse=float)


@task
def get_dataset(test_size: float, random_state: int) -> DatasetSplits:
    dataset = fetch_california_housing(as_frame=True).frame
    return train_test_split(dataset, test_size=test_size, random_state=random_state)


@task
def summarize_dataset(dataset: Dataset) -> pd.DataFrame:
    return dataset.describe()


@task
def train_model(dataset: Dataset, hyperparameters: Hyperparameters) -> Ridge:
    model = Ridge(**asdict(hyperparameters))
    return model.fit(dataset.drop(TARGET, axis="columns"), dataset[TARGET])


@task
def evaluate_model(dataset: Dataset, model: Ridge) -> float:
    features, target = dataset.drop(TARGET, axis="columns"), dataset[TARGET]
    return mean_squared_error(target, model.predict(features))


@workflow
def main(hyperparameters: Hyperparameters, test_size: float = 0.2, random_state: int = 43) -> TrainingResult:
    train_dataset, test_dataset = get_dataset(test_size=test_size, random_state=random_state)

    summarize_dataset(dataset=train_dataset)
    summarize_dataset(dataset=test_dataset)

    model = train_model(dataset=train_dataset, hyperparameters=hyperparameters)
    train_mse = evaluate_model(dataset=train_dataset, model=model)
    test_mse = evaluate_model(dataset=test_dataset, model=model)

    return model, train_mse, test_mse


if __name__ == "__main__":
    print(f"trained model: {main(hyperparameters=Hyperparameters(alpha=10.0))}")
