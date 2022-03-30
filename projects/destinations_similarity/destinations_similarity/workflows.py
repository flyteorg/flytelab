"""Workflows for the destinations_similarity Flyte project."""

import json
import os
from datetime import timedelta

import pandas as pd
from flytekit import workflow, LaunchPlan, FixedRate

from destinations_similarity import tasks

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

CONFIG = json.load(open(os.path.join(CURRENT_DIRECTORY,'config.json')))

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
def build_knowledge_base(remote_dataset: str = "") -> None:
    """Retrieve knowledge base

    Args:
        remote_dataset (str): Remote dataset's URL. Defaults to ''.

    """

    dataframe = tasks.retrieve_dataset_from_remote(remote_dataset)
    
    dataframe = tasks.preprocess_input_data(dataframe,CONFIG['columns_to_translate'], CONFIG['columns_to_process'])

    list_dataframes = tasks.vectorize_columns(dataframe,CONFIG['columns_to_vec'],CONFIG['city_column_name'])

    final_city_vectors = tasks.build_mean_embedding(list_dataframes)

    ######## final_city_vectors: SAVE DATAFRAME INTO CSV OR OTHER FORMAT ###########

@workflow
def inference(dataframe: pd.DataFrame, dataframe_vectorized: pd.DataFrame, 
                    kneighborhood:int, vector_dim: int, actual_city_name:str,
                    see_wikivoyage,do_wikivoyage,city_column) -> dict:
    """Retrieve knowledge base

    Args:

    Returns:
        
    """
    
    nearst_city = tasks.get_k_most_near(dataframe_vectorized,kneighborhood,
                                                vector_dim,actual_city_name,city_column)

    output = tasks.build_output(dataframe,nearst_city,see_wikivoyage,
                                do_wikivoyage,city_column,kneighborhood,actual_city_name)
    
    
    return output                                       


# Launch plans
build_knowledge_base = LaunchPlan.get_or_create(
    name='build_knowledge_base',
    workflow=build_knowledge_base,
    default_inputs={
        'remote_dataset':
        "https://storage.googleapis.com"
        "/dsc-public-info/datasets/flytelab_dataset.parquet",
    },
    schedule=FixedRate(duration=timedelta(weeks=4))
)
