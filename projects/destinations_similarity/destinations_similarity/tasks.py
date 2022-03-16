"""Tasks for the destinations_similarity Flyte project."""

import os
import sys
import logging

import pandas as pd
from flytekit import task, Resources
from sklearn.metrics import r2_score

from destinations_similarity import utils


# Logging config
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(levelname)s | %(message)s"
)


@task(cache=False, requests=Resources(cpu="1", mem="500Mi"))
def get_dataset_from_bucket(bucket: str, file_path: str) -> pd.DataFrame:
    """Retrieve dataset from GCS bucket and return it as a DataFrame.

    Args:
        bucket (str): The bucket name.
        file_path (str): The path of the file on the bucket.

    Returns:
        pd.DataFrame: The dataset.
    """
    # Retrieving file
    tmp_path = "/tmp/flyte/dataset.blob"
    gcloud = utils.google.GoogleUtils()
    gcloud.download_blob(bucket, file_path, tmp_path)

    # Converting to pandas, converting columns to strs
    result = pd.read_csv(tmp_path)
    result.columns = result.columns.astype(str)

    LOGGER.info("Dataset gs://%s/%s retrieved.", bucket, file_path)
    os.remove(tmp_path)
    return result


@task(cache=True, cache_version='1.0')
def score_prediction(
    y_true: pd.DataFrame, y_pred: pd.DataFrame,
) -> float:
    """Compute the R2 score from true and predicted values.

    Args:
        y_true (DataFrame): The true values.
        y_pred (DataFrame): The predicted values.

    Returns:
        score (float): The R2 score.
    """
    return r2_score(y_true, y_pred)
