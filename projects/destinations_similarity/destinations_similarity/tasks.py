"""Tasks for the destinations_similarity Flyte project."""

import sys
import logging
from typing import List, Dict

import requests
import pandas as pd
from flytekit import task, Resources

from destinations_similarity.scraper.extractor import WikiExtractor
from destinations_similarity.scraper.brazilian_cities import (
    get_brazilian_cities_data, get_dataframe
)


# Logging config
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(levelname)s | %(message)s"
)

# Flyte configuration
BASE_RESOURCES = Resources(cpu="0.5", mem="500Mi")


@task(retries=3, requests=BASE_RESOURCES)
def get_base_data(generate_city_id: bool) -> pd.DataFrame:
    """Retrieve base data for the dataset.

    Args:
        generate_city_id (bool): Informs if an ID must be generated for each
            row of the dataset.

    Returns:
        pd.DataFrame: Base dataset.
    """
    return get_brazilian_cities_data(
        get_dataframe, generate_city_id=generate_city_id)


@task(retries=3, requests=BASE_RESOURCES)
def scrap_wiki(
    base_data: pd.DataFrame, wiki: str, lang: str, summary: bool,
    sections: List[str], sections_tags: Dict[str, List[str]],
    sections_types: Dict[str, str]
) -> pd.DataFrame:
    """Scrap a Wikimedia page for info.

    Args:
        base_data (pd.DataFrame): Base dataset.
        wiki (str): Type of wiki ('wikipedia', 'wikivoyage').
        lang (str): Language of wiki.
        summary (bool): If the summary must be retrieved.
        sections (List[str]): Which sections must be retrieved.
        sections_tags (Dict[str, List[str]]): Which HTML tags must be preserved
            for a given section.
        sections_types (Dict[str, str]): How each section will be
            saved on the dataset, 'str' or 'list'.

    Returns:
        pd.DataFrame: The updated dataset.
    """
    # Initialize scraper
    extractor = WikiExtractor(wiki=wiki, lang=lang)

    # Setup fields for the sections
    sections_fields = {
        section: f"{section}_{wiki}_{lang}".lower().replace(' ', '_')
        for section in ['summary'] + sections
    }

    # Initialize dataset
    dataset = base_data.copy()
    dataset[f"images_{wiki}_{lang}"] = [[] for _ in range(len(dataset))]
    for section, field in sections_fields.items():
        dataset[field] = (
            [[] for _ in range(len(dataset))]
            if sections_types.get(section) == 'list' else ""
        )

    # Retrieve data for each city
    for i, row in dataset.iterrows():
        page_name = row[f"title_{wiki}_{lang}"]

        # Set content
        page_content = extractor.extract_content(
            page_name, summary=summary, sections=sections,
            sections_tags=sections_tags, section_types=sections_types
        )
        for section, text in page_content.items():
            dataset.at[i, sections_fields[section]] = text

        # Set images links
        page_images = extractor.extract_images(page_name)
        dataset.at[i, f"images_{wiki}_{lang}"] = page_images

    return dataset


@task(cache=True, cache_version='1.0', requests=BASE_RESOURCES)
def merge_dataframes(
    df_x: pd.DataFrame, df_y: pd.DataFrame, join: str
) -> pd.DataFrame:
    """Merge two DataFrames together.

    Args:
        df_x (pd.DataFrame): First DataFrame.
        df_y (pd.DataFrame): Second DataFrame.
        join (str): The type of merge, 'inner' or 'outer'.

    Returns:
        pd.DataFrame: The concatenation of the DataFrames.
    """
    df_y_columns = df_y.columns.difference(df_x.columns)
    return pd.concat([df_x, df_y[df_y_columns]], axis=1, join=join)


@task(retries=3, requests=BASE_RESOURCES)
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
