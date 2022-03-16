"""Workflows for the destinations_similarity Flyte project."""

from flytekit import workflow

from destinations_similarity import tasks


@workflow
def train_model() -> dict:
    """Train a clusterization model from a dataset."""
    _ = tasks.get_dataset_from_bucket(
        bucket='dsc_public', file_path='my/file/path.csv')


if __name__ == "__main__":
    print(f"trained model: {train_model()}")
