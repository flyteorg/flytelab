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
                    kneighborhood:int, vector_dim: int, actual_city_name:str) -> dict:
    """Retrieve knowledge base

    Args:

    Returns:
        
    """
    
    see_wikivoyage = CONFIG['see_wikivoyage_column_name']
    do_wikivoyage = CONFIG['do_wikivoyage_column_name']
    city_column = CONFIG['city_column_name']

    nearst_city = tasks.get_k_most_near(dataframe_vectorized,kneighborhood,
                                                vector_dim,actual_city_name,city_column)
    
    pois_target = dataframe[dataframe[city_column]==actual_city_name][[see_wikivoyage, do_wikivoyage]]
    
    if len(pois_target)>0:
        to_see_actual = tasks.translate_description(pois_target[see_wikivoyage].iloc[0],'en')
        to_do_actual = tasks.translate_description(pois_target[do_wikivoyage].iloc[0],'en')
    else:
        to_see_actual = "\nOops..unfortunately we don't have the record of what Kin did in this city :/\n"
        to_do_actual = "\nOops..unfortunately we don't have the record of what Kin did in this city :/\n"
    
    
    to_see_nearst = []
    to_do_nearst = []
    for cits in range(kneighborhood):
        pois_suggestion = dataframe[dataframe[city_column]==nearst_city.iloc[cits]][[see_wikivoyage, do_wikivoyage]]
        if len(pois_suggestion)>0:
            to_see_nearst.append(tasks.translate_description(pois_suggestion[see_wikivoyage].iloc[0],'en'))
            to_do_nearst.append(tasks.translate_description(pois_suggestion[do_wikivoyage].iloc[0],'en'))
        else:
            to_see_nearst.append("\nOops..unfortunately we don't have information about this city :/\n")
            to_do_nearst.append("\nOops..unfortunately we don't have information about this city :/\n")

    output = {
        "actual_city": actual_city_name,
        "actual_city_to_see":to_see_actual,
        "actual_city_to_do":to_do_actual,
        "nearst_city":list(nearst_city),
        "nearst_to_see":to_see_nearst,
        "nearst_to_see":to_do_nearst
    }

    return output                                       



@workflow
def inference(dataframe: pd.DataFrame, dataframe_vectorized: pd.DataFrame, 
                    kneighborhood:int, vector_dim: int, city_name:str) -> None:
    """Retrieve knowledge base

    Args:
        remote_dataset (str): Remote dataset's URL. Defaults to ''.

    Returns:

    """
    
    see_wikivoyage = CONFIG['see_wikivoyage_column_name']
    do_wikivoyage = CONFIG['do_wikivoyage_column_name']
    city_column = CONFIG['city_column_name']

    nearst_city = tasks.get_k_most_near(dataframe_vectorized,kneighborhood,
                                                vector_dim,city_name,city_column)
    
    pois_target = dataframe[dataframe[city_column]==city_name][[see_wikivoyage, do_wikivoyage]]

    print('Last city visited: {}\n'.format(city_name))
    if len(pois_target)>0:
        print('What you saw in the last city:\n{}\n'.format(tasks.translate_description(pois_target[see_wikivoyage].iloc[0],'en')))
        print('What you did in the last city:\n{}\n'.format(tasks.translate_description(pois_target[do_wikivoyage].iloc[0],'en')))
    else:
        print("\nOops..unfortunately we don't have the record of what Kin did in this city :/\n")
    
    for cits in range(kneighborhood):
        pois_suggestion = dataframe[dataframe[city_column]==nearst_city.iloc[cits]][[see_wikivoyage, do_wikivoyage]]
        print('\nSuggestion of next cities to visit: {}\n'.format(nearst_city.iloc[cits]))
        if len(pois_suggestion)>0:
            print('\nWhat you will see in the next cities\n{}'.format(tasks.translate_description(pois_suggestion[see_wikivoyage].iloc[0],'en')))
            print('\nWhat you will do in the next cities\n{}'.format(tasks.translate_description(pois_suggestion[do_wikivoyage].iloc[0],'en')))
        else:
            print("\nOops..unfortunately we don't have information about this city :/\n")                                           


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
