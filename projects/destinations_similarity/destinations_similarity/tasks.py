"""Tasks for the destinations_similarity Flyte project."""

import sys
import logging
from typing import List, Dict, Tuple

import torch
import pandas as pd
import numpy as np
from unidecode import unidecode
from flytekit import task, Resources
from flytekit.types.file import FlyteFile

from destinations_similarity.scraper.extractor import WikiExtractor
from destinations_similarity.scraper.brazilian_cities import (
    get_brazilian_cities_data, get_dataframe)
from destinations_similarity.processing.text_preprocessing import (
    translate_description_series, preprocess_text)
from destinations_similarity.processing.feature_engineering import (
    TextVectorizer)


# Logging config
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(levelname)s | %(message)s"
)

# Flyte configuration
BASE_RESOURCES = Resources(cpu="0.5", mem="500Mi")
INTENSIVE_RESOURCES = Resources(cpu="1", mem="1Gi", gpu="1")


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


@task(cache=True, cache_version='1.0')
def check_if_remote(uri: str) -> Tuple[bool, FlyteFile]:
    """Check if a URI points to a remote file."""
    if uri:
        return True, uri
    return False, uri


@task(retries=3, requests=BASE_RESOURCES)
def retrieve_dataset_from_remote(uri: FlyteFile) -> pd.DataFrame:
    """Retrieve a dataset from a remote URL.

    Args:
        url (FlyteFile): Remote address of the dataset. Must be a Parquet file.

    Returns:
        pd.DataFrame: DataFrame with the dataset.
    """
    # Download file if it has a remote source
    if uri.remote_source is not None:
        uri.download()

    dataset_df = pd.read_parquet(uri.path)
    dataset_df.columns = dataset_df.columns.astype(str)
    LOGGER.info("Retrieved dataset from '%s'.", uri.remote_source or uri.path)
    return dataset_df


@task(cache=True, cache_version='1.0', requests=INTENSIVE_RESOURCES)
def preprocess_input_data(
    dataframe: pd.DataFrame, columns_to_translate: List[str],
    columns_to_process: List[str], wikivoyage_summary: str
) -> pd.DataFrame:
    """Preprocess the scraped data.

    Args:
        dataframe (pd.DataFrame): remote dataframe with cities features
        columns_to_translate (List[str]): city features to be translated
        columns_to_process (List[str]): city features to be processed
        wikivoyage_summary (str): summary wikivoyage column name

    Returns:
        pd.DataFrame: remote dataframe pre-processed
    """
    LOGGER.info("Preprocessing input data.")

    if wikivoyage_summary:
        dataframe = dataframe[
            dataframe[wikivoyage_summary].notna()
        ].copy().reset_index(drop=True)
        LOGGER.info("Using %s rows of data.", dataframe.shape[0])

    # Translate columns
    for col in columns_to_translate:
        dataframe[col] = translate_description_series(dataframe, col)

    LOGGER.info("Columns %s translated.", columns_to_translate)

    # Process specified columns
    for col in columns_to_process:
        dataframe[col] = dataframe[col].fillna("").swifter.apply(
            lambda x: unidecode(x) if isinstance(x, str) else x).str.lower()
        dataframe[col] = preprocess_text(dataframe, col)

    LOGGER.info("Columns %s processed.", columns_to_process)
    dataframe.columns = dataframe.columns.astype(str)
    return dataframe


@task(cache=True, cache_version='1.0', requests=INTENSIVE_RESOURCES)
def vectorize_columns(
    dataframe: pd.DataFrame, columns_to_vec: List[str],
    city_column: str, state_column: str
) -> List[pd.DataFrame]:
    """Generate embeddings with the cities' infos.

    Args:
        dataframe (pd.DataFrame): remote dataset pre-processed
        columns_to_vec (List[str]): city features to be vectorized
        city_column (str): city column name
        state_column (str): state column name

    Returns:
        List[pd.DataFrame]: list of dataframes with city feature vectors
    """
    model = TextVectorizer()
    column_embeddings = []

    LOGGER.info("Generating embeddings for columns.")

    # Generate embeddings for each column
    for col in columns_to_vec:
        inputs_ids = model.encode_inputs(dataframe[col])
        embeddings = model.get_df_embedding(inputs_ids)
        city_embeddings = pd.concat(
            [dataframe[[city_column, state_column]], embeddings], axis=1)
        city_embeddings.columns = city_embeddings.columns.astype(str)
        column_embeddings.append(city_embeddings)

    LOGGER.info("Embeddings generated.")
    return column_embeddings


@task(cache=True, cache_version='1.0', requests=INTENSIVE_RESOURCES)
def build_mean_embedding(
    list_dataframes: List[pd.DataFrame]
) -> pd.DataFrame:
    """Build mean embeddings for cities.

    Args:
        list_dataframes (List[pd.DataFrame]): list of dataframes with
            city feature vectors

    Returns:
        pd.DataFrame: city vectors
    """
    LOGGER.info("Building mean embeddings.")

    # Retrieve embeddings
    column_embeddings = [data.iloc[:, 2:].values for data in list_dataframes]

    # Compute mean embeddings
    aux = torch.Tensor(np.array(column_embeddings))
    aux_mean = aux.mean(axis=0)
    aux_mean = pd.DataFrame(aux_mean).astype("float")
    aux_mean = aux_mean.fillna(0)
    aux_mean = pd.concat(
        [list_dataframes[0][['city', 'state']], aux_mean], axis=1)
    aux_mean.columns = aux_mean.columns.astype(str)

    LOGGER.info("Mean embeddings calculated.")
    return aux_mean
