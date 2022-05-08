import json
import typing
import warnings
from train import Hyperparameters, train
from flytekit import Resources, task, workflow
from preprocess import clean_dataset, preprocess
from flytekit.types.directory import FlyteDirectory
from datasource import download_gtzan_repo, GTZAN_ZIP_FILE_PATH


SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
warnings.filterwarnings("ignore")
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

MODELSAVE = [typing.TypeVar("str")]
workflow_outputs = typing.NamedTuple("WorkflowOutputs", model=FlyteDirectory[MODELSAVE])


@task
def download_gtzan_dataset():
    download_gtzan_repo()


@task
def clean_gtzan_dataset():
    clean_dataset()


@task(cache_version="1.0", cache=True, limits=Resources(mem="800Mi"))
def preprocess_gtzan_dataset(
    dataset_path: str
) -> dict:
    processed_data = preprocess(dataset_path=dataset_path)
    return processed_data


@task(cache_version="1.0", cache=True, limits=Resources(mem="800Mi"))
def train_gtzan_dataset(
    data: dict,
    hp: Hyperparameters,
):
    model = train(data=data, hp=hp)
    return model


@workflow
def flyteworkflow(
    dataset_path: str = GTZAN_ZIP_FILE_PATH
):
    download_gtzan_dataset()
    clean_gtzan_dataset()
    processed_data = preprocess_gtzan_dataset(
        dataset_path=dataset_path,
    )
    model = train_gtzan_dataset(
        data=processed_data,
        hp=Hyperparameters(epochs=10)
    )

    # return (model.model)


if __name__ == "__main__":
    print(f"Running {__file__} main...")
    flyteworkflow()
