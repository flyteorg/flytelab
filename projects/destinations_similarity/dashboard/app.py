"""Module for the Streamlit app."""

# pylint: disable=no-value-for-parameter,broad-exception

import os
import sys
import logging
from argparse import ArgumentParser
from pathlib import Path
from tabnanny import verbose
from io import BytesIO

from typing import List
import faiss
import requests
import streamlit as st
import pandas as pd
import numpy as np
from flytekit.remote import FlyteRemote
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from PIL import Image
from deep_translator import GoogleTranslator


PROJECT_NAME = "vamos-dalhe"
WORKFLOW_NAME = "build_knowledge_base_default_lp"

# Logging config
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(levelname)s | %(message)s"
)

def retrieve_dataset_from_remote(url: str) -> pd.DataFrame:
    """Retrieve the dataset from a remote URL.
    Args:
        url (str): Remote address of the dataset, a Parquet file.
    Returns:
        pd.DataFrame: DataFrame with the dataset.
    """
    dataset_parquet = requests.get(url, timeout=30)
    dataset_df = pd.read_parquet(dataset_parquet.content)
    LOGGER.info("Retrieved dataset from %s.", url)
    return dataset_df

def get_k_nearest_neighbors(
    embeddings: pd.DataFrame, k_neighbors: int, city_name: str, state_name: str
) -> pd.DataFrame:
    """Retrieve the k-nearest neighbors of a city.
    Args:
        embeddings (pd.DataFrame): city vectors
        k_neighbors (int): number os similar cities to present
        city_name (str): last city visited
        state_name (str): last state visited
    Returns:
        pd.DataFrame: the cities most similar to city_name
    """
    # Retrieve vectors to search
    vec_name = embeddings[~(
        (embeddings['city'] == city_name) & (embeddings['state'] == state_name)
    )].reset_index(drop=True)
    vec = vec_name.drop(['city', 'state'], axis=1)

    # Initialize faiss
    index = faiss.IndexFlatL2(vec.shape[1])
    index.add(np.ascontiguousarray(np.float32(vec.values)))

    # Build query
    query = embeddings[(
        (embeddings['city'] == city_name) & (embeddings['state'] == state_name)
    )].drop(['city', 'state'], axis=1).values
    query = np.float32(query)

    # Retrieve k-nearest neighbors
    _, indexes = index.search(query, k_neighbors)
    nearest = vec_name[['city', 'state']].iloc[indexes[0]]

    return nearest


def translate_description(text: str, target_lang: str = 'pt') -> str:
    """Translate non-portuguese text.
    Args:
        text (str): column name to be translated
        target_lang (str): taget language
    Returns:
        str: text translated
    """
    try:
        return GoogleTranslator(
            source='auto', target=target_lang).translate(text)
    except Exception:
        return text


def build_output(
    dataset: pd.DataFrame, nearest_cities: pd.DataFrame, columns_to_retrieve: List[str]
) -> pd.DataFrame:
    """_summary_
    Args:
        dataset (pd.DataFrame): dataset scrapper from wikipedia and wikivoyage
        nearest_cities (pd.DataFrame): output model of the nearest cities
        columns_to_retrieve (List[str]): list of columns to add to output
    Returns:
        output (pd.DataFrame): _description_
    """
    
    output = dataset.merge(nearest_cities, on=['city', 'state'])

    output = output[['city', 'state', columns_to_retrieve[0], columns_to_retrieve[1]]]

    return output


if __name__ == '__main__':
    # Retrieve arguments
    parser = ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    backend = os.getenv(
        "FLYTE_BACKEND", 'remote' if args.remote else 'sandbox')


    if args.offline:
        # Configuration for accessing a Flyte cluster backend
        remote = FlyteRemote.from_config(
            default_project=PROJECT_NAME,
            default_domain="development",
            config_file_path=Path(__file__).parent / f"{backend}.config",
        )

        # Get the latest workflow execution
        [latest_execution, *_], _ = remote.client.list_executions_paginated(
            PROJECT_NAME,
            "development",
            limit=1,
            filters=[
                filters.Equal("launch_plan.name", WORKFLOW_NAME),
                filters.Equal("phase", "SUCCEEDED"),
            ],
            sort_by=Sort.from_python_std("desc(execution_created_at)"),
        )

        wf_execution = remote.fetch_workflow_execution(name=latest_execution.id.name)
        remote.sync(wf_execution, sync_nodes=False)
        embeddings = wf_execution.outputs["o0"]
        print(embeddings)
    else:
       # embeddings = pd.read_parquet('./embeddings.parquet')
        embeddings = pd.read_parquet("https://storage.googleapis.com/dsc-public-info/datasets/flytelab_embeddings.parquet")
        print("Embeddings loaded")
    
    wiki_dataset = pd.read_parquet("https://storage.googleapis.com/dsc-public-info/datasets/flytelab_dataset.parquet")

    st.write("# Flytelab: destinations_similarity")
    
    st.write('#### Kinder is an adventurous dog who loves to travel! He enjoys specially nature places: beachs, waterfalls, trails and more, which Brazil is not missing. He wants experiences in other cities but he doesnt know where.')
    st.write("## So he is now asking, **where should I go next**?")

    beach_kinder = Image.open('beach_kinder.jpeg')

    st.image(beach_kinder, caption='Kinder in love with the beach')

    st.write("Help Kinder by selecting a city you like in Brazil below so we can recommend similar places he will most certainly enjoy!")

    desired_state = st.selectbox('From state...', wiki_dataset['state'].unique().tolist())
    desired_city = st.selectbox('I like the city:', wiki_dataset[wiki_dataset['state'] == desired_state]['city'].unique().tolist())

    n_cities = st.slider('How many recommendations do you want?', 1, 30, 5)

    cities_recommended = get_k_nearest_neighbors(
        embeddings=embeddings, k_neighbors=n_cities,
        city_name=desired_city, state_name=desired_state
    )

    st.write("## So, where next?")
    st.dataframe(build_output(
        dataset=wiki_dataset, nearest_cities=cities_recommended,
        columns_to_retrieve=['see_wikivoyage_en', 'do_wikivoyage_en']
    ))

    st.write("Hope you enjoy the recommendation! See you on your next trip.")

    kinder = Image.open('kinder.jpeg')

    st.image(kinder, caption='The marvelous Kinder')


