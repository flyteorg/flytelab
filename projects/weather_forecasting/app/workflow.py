import os
from datetime import datetime

from flytekit import task, workflow

from flytelab.weather_forecasting import data


DEFAULT_DATE = datetime.now().date()


@task
def fetch_key(key: str) -> str:
    print("fetching API key")
    return os.getenv(key)


@workflow
def get_api_key(key: str) -> str:
    return fetch_key(key=key)


@task
def get_training_instance(location: str, target_date: datetime) -> data.TrainingInstance:
    return data.get_training_instance(location, target_date.date())


@workflow
def run_pipeline(location: str = "Atlanta, GA US", target_date: datetime = datetime.now()) -> data.TrainingInstance:
    return get_training_instance(location=location, target_date=target_date)


if __name__ == "__main__":
    run_pipeline()
