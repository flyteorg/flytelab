import logging
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
from typing import NamedTuple

import flytekitplugins.pandera  # noqa
import pandas as pd
import pandera as pa
import scipy.stats as stats
from pandera.typing import DataFrame, Index, Series
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from flytekit import task, workflow

from california_housing_regression import custom_checks


logger = logging.getLogger("flytekit")
logger.setLevel(logging.ERROR)


@dataclass_json
@dataclass
class Hyperparameters:
    alpha: float
    random_state: int = 42


class CaliforniaHousingData(pa.SchemaModel):
    Latitude: Series[float] = pa.Field(in_range={"min_value": -90, "max_value": 90})
    Longitude: Series[float] = pa.Field(in_range={"min_value": -180, "max_value": 180})
    AveBedrms: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1_000_000})
    AveOccup: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1_000_000})
    AveRooms: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1_000_000})
    HouseAge: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1_000_000})
    MedInc: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1_000_000})
    MedHouseVal: Series[float] = pa.Field(
        mean_eq={
            "value": 2.0685,
            "alpha": 1e-3,
            "error": "MedHouseVal mean value is not equal to 2.0685 [alpha=1e-3]",
        }
    )

    class Config:
        coerce = True


class CaliforniaHousingDataSummary(CaliforniaHousingData):
    Latitude: Series[float]
    Longitude: Series[float]
    AveBedrms: Series[float]
    AveOccup: Series[float]
    AveRooms: Series[float]
    HouseAge: Series[float]
    MedInc: Series[float]
    MedHouseVal: Series[float]
    
    index: Index[str] = pa.Field(eq=["count", "mean", "std", "min", "25%", "50%", "75%", "max"])


Dataset = DataFrame[CaliforniaHousingData]
DatasetSummary = DataFrame[CaliforniaHousingDataSummary]


TARGET = "MedHouseVal"

DatasetSplits = NamedTuple("DatasetSplits", train=Dataset, test=Dataset)
TrainingResult = NamedTuple("TrainingResult", model=Ridge, train_mse=float, test_mse=float)


@task
def get_dataset(test_size: float, random_state: int) -> DatasetSplits:
    dataset = fetch_california_housing(as_frame=True).frame
    return train_test_split(dataset, test_size=test_size, random_state=random_state)


@task
def summarize_dataset(dataset: Dataset) -> DatasetSummary:
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
