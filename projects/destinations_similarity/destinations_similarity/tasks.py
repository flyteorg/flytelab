"""Tasks for the destinations_similarity Flyte project."""

import sys
import logging

import requests
import pandas as pd
from flytekit import task, Resources


# Logging config
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(levelname)s | %(message)s"
)

# Flyte configuration
BASE_RESOURCES = Resources(cpu="0.5", mem="500Mi")


@task(cache=False, requests=BASE_RESOURCES)
def retrieve_dataset_from_remote(url: str) -> pd.DataFrame:
    """Retrieve the dataset from a remote URL.

    Args:
        url (str): Remote address of the dataset, a Parquet file.

    Returns:
        pd.DataFrame: DataFrame with the dataset.
    """
    dataset_parquet = requests.get(url, timeout=30)
    dataset_df = pd.read_parquet(dataset_parquet)
    LOGGER.info("Retrieved dataset from %s.", url)
    return dataset_df
