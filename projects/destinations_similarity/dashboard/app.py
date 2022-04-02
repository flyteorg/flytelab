"""Module for the Streamlit app."""

# pylint: disable=no-value-for-parameter

import os
import sys
import logging
from argparse import ArgumentParser

from typing import List
import faiss
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


GCS_BUCKET_PATH = "https://storage.googleapis.com/dsc-public-info/datasets/"
EMBEDDINGS_FILENAME = "flytelab_embeddings.parquet"
DATASET_FILENAME = "flytelab_dataset.parquet"


# Logging config
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(levelname)s | %(message)s"
)


def retrieve_dataframe_from_remote(dataset_name: str) -> pd.DataFrame:
    """Retrieve a dataset saved as Parquet from remote."""
    return pd.read_parquet(GCS_BUCKET_PATH + dataset_name)


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


def build_output(
    dataset: pd.DataFrame, nearest_cities: pd.DataFrame,
    columns_to_retrieve: List[str]
) -> pd.DataFrame:
    """Build the output text of inference.

    Args:
        dataset (pd.DataFrame): dataset scraper from wikipedia and wikivoyage
        nearest_cities (pd.DataFrame): output model of the nearest cities
        columns_to_retrieve (List[str]): list of columns to add to output

    Returns:
        str: Markdown-formatted text
    """
    output = ""
    default_desc = (
        "\nOops... Unfortunately we don't have records for this city. "
        "\U0001F615\n"
    )

    for _, row in nearest_cities.iterrows():
        output += f"\n## {row.city}, {row.state}\n"

        pois_suggestion = dataset[
            (dataset['city'] == row.city) & (dataset['state'] == row.state)
        ][columns_to_retrieve].iloc[0]

        for column in columns_to_retrieve:
            section = ' '.join(column.split('_')[:-2]).capitalize()
            output += (
                f"\n### {section}"
                f"\n{pois_suggestion[column] or default_desc}")

    return output


if __name__ == '__main__':
    # Retrieve arguments
    parser = ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    backend = os.getenv(
        "FLYTE_BACKEND", 'remote' if args.remote else 'sandbox')

    # Retrieve datasets from remote
    embs_df = retrieve_dataframe_from_remote(EMBEDDINGS_FILENAME)
    wiki_df = retrieve_dataframe_from_remote(DATASET_FILENAME)

    # App definition
    st.write(
        "# Flytelab: Destinations Similarity\n"
        "Kinder is an adventurous dog who loves to travel! He enjoys "
        "specially nature places: beaches, waterfalls, trails and more, "
        "which Brazil surely is abundant of. He wants experiences in other "
        "cities but he doesn't know where to go.\n"
        "## So he is now asking, **where should I go next**?"
    )

    beach_kinder = Image.open('beach_kinder.jpeg')
    st.image(beach_kinder, caption='Kinder in love with the beach')

    st.write(
        "Help Kinder by selecting a city you like in Brazil below so we can "
        "recommend similar places that he will most certainly enjoy!"
    )

    # Select city, state, and n of recommendations
    desired_state = st.selectbox(
        'From state...',
        embs_df['state'].unique().tolist()
    )
    desired_city = st.selectbox(
        'I like the city:',
        embs_df[embs_df['state'] == desired_state]['city'].unique().tolist()
    )

    n_cities = st.slider('How many recommendations do you want?', 1, 30, 5)

    # Get recommendations
    cities_recommended = get_k_nearest_neighbors(
        embeddings=embs_df, k_neighbors=n_cities,
        city_name=desired_city, state_name=desired_state
    )

    st.write("## So, where next?")
    st.write(build_output(
        dataset=wiki_df, nearest_cities=cities_recommended,
        columns_to_retrieve=[
            'summary_wikivoyage_en'
        ]
    ))

    kinder = Image.open('kinder.jpeg')
    st.image(kinder, caption='The marvelous Kinder')

    st.write(
        "We hope you enjoy the recommendations! See you on your next trip."
    )
