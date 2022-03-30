"""Workflows for the destinations_similarity Flyte project."""

from typing import Dict
from datetime import timedelta

import pandas as pd
from flytekit import workflow, conditional, LaunchPlan, FixedRate

from destinations_similarity import tasks


@workflow
def generate_dataset() -> pd.DataFrame:
    """Generate the dataset to be used for training.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    base_data = tasks.get_base_data(generate_city_id=True)

    # Retrieve data from pt.wikipedia.org
    data_wikipedia_pt = tasks.scrap_wiki(
        base_data=base_data, wiki='wikipedia', lang='pt', summary=True,
        sections=['Clima', 'Economia', 'História', 'Geografia'],
        sections_tags={}, sections_types={}
    )

    # Retrieve data from en.wikivoyage.org
    data_wikivoyage_en = tasks.scrap_wiki(
        base_data=base_data, wiki='wikivoyage', lang='en', summary=True,
        sections=['Do', 'See', 'Go next'],
        sections_tags={'Go next': ['a', 'b']},
        sections_types={'Go next': 'list'}
    )

    # Merge data
    dataset = tasks.merge_dataframes(
        df_x=data_wikipedia_pt, df_y=data_wikivoyage_en, join='outer')

    return dataset


@workflow
def train_model(remote_dataset: str = "") -> Dict[str, pd.DataFrame]:
    """Retrieve dataset and train model.

    Args:
        dataset_url (str, optional): Remote dataset's URL. Defaults to ''.

    Returns:
        dict: Information about the model.
    """
    dataset = (
        conditional("remote_dataset")
        .if_(remote_dataset != "")
        .then(tasks.retrieve_dataset_from_remote(url=remote_dataset))
        .else_()
        .then(generate_dataset())
    )

    # TODO: Train model

    return {'dataset': dataset}


# Launch plans

train_model_lp = LaunchPlan.get_or_create(
    name='train_model_lp',
    workflow=train_model,
    default_inputs={
        'remote_dataset':
        "https://storage.googleapis.com"
        "/dsc-public-info/datasets/flytelab_dataset.parquet",
    },
    schedule=FixedRate(duration=timedelta(weeks=4))
)
