"""Workflows for the destinations_similarity Flyte project."""

import os
import json
from datetime import timedelta
from typing import List

import pandas as pd
from flytekit import workflow, conditional, LaunchPlan, FixedRate

from destinations_similarity import tasks


# Retrieve configuration file
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIRECTORY, 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as file_desc:
    CONFIG = json.load(file_desc)


@workflow
def generate_dataset() -> pd.DataFrame:
    """Generate the dataset to be used for training.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    base_data = tasks.get_base_data(generate_city_id=False)

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
def build_knowledge_base(
    columns_to_translate: List[str], columns_to_process: List[str],
    summary_wikivoyage_column_name: str, remote_dataset: str = ""
) -> pd.DataFrame:
    """Generate knowledge database.

    Args:
        columns_to_translate (List[str]): _description_
        columns_to_process (List[str]): _description_
        summary_wikivoyage_column_name (str): _description_
        remote_dataset (str, optional): Remote dataset's URL. Generates
            dataset if no path is specified.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    remote, flyte_file = tasks.check_if_remote(uri=remote_dataset)

    dataframe = (
        conditional("remote_dataset")
        .if_(remote.is_true())      # pylint: disable=no-member
        .then(tasks.retrieve_dataset_from_remote(uri=flyte_file))
        .else_()
        .then(generate_dataset())
    )

    dataframe_processed = tasks.preprocess_input_data(
        dataframe=dataframe,
        columns_to_translate=columns_to_translate,
        columns_to_process=columns_to_process,
        wikivoyage_summary=summary_wikivoyage_column_name
    )

    list_dataframes = tasks.vectorize_columns(
        dataframe=dataframe_processed,
        columns_to_vec=columns_to_process,
        city_column='city',
        state_column='state'
    )

    city_vectors = tasks.build_mean_embedding(list_dataframes=list_dataframes)

    return city_vectors


@workflow
def inference(
    dataframe: pd.DataFrame, dataframe_vectorized: pd.DataFrame,
    k_neighbors: int, city_name: str, state_name: str,
    see_wikivoyage_column: pd.DataFrame, do_wikivoyage_column: pd.DataFrame
) -> dict:
    """Infer data.

    Args:
        dataframe (pd.DataFrame): _description_
        dataframe_vectorized (pd.DataFrame): _description_
        k_neighbors (int): _description_
        vector_dim (int): _description_
        actual_city_name (str): _description_
        see_wikivoyage (pd.DataFrame): _description_
        do_wikivoyage (pd.DataFrame): _description_
        city_column (str): _description_

    Returns:
        dict: _description_
    """
    nearest_cities = tasks.get_k_nearest(
        embeddings=dataframe_vectorized,
        k_neighbors=k_neighbors,
        city_name=city_name,
        state_name=state_name
    )

    output = tasks.build_output(
        dataframe=dataframe,
        nearest_cities=nearest_cities,
        see_wikivoyage_column=see_wikivoyage_column,
        do_wikivoyage_column=do_wikivoyage_column,
        city_name=city_name,
        state_name=state_name
    )

    return output


# Launch plans
build_knowledge_base_lp = LaunchPlan.get_or_create(
    name='build_knowledge_base_lp',
    workflow=build_knowledge_base,
    default_inputs={
        'columns_to_translate': [
            "see_wikivoyage_en",
            "do_wikivoyage_en",
            "summary_wikivoyage_en"
        ],
        'columns_to_process': [
            "summary_wikipedia_pt",
            "história_wikipedia_pt",
            "geografia_wikipedia_pt",
            "clima_wikipedia_pt",
            "see_wikivoyage_en",
            "do_wikivoyage_en",
            "summary_wikivoyage_en"
        ],
        'summary_wikivoyage_column_name': "summary_wikivoyage_en",
        'remote_dataset':
        "https://storage.googleapis.com"
        "/dsc-public-info/datasets/flytelab_dataset.parquet",
    },
    schedule=FixedRate(duration=timedelta(weeks=4))
)
