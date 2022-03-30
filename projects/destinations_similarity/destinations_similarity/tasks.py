"""Tasks for the destinations_similarity Flyte project."""

import sys
import logging
from typing import List, Dict

import requests
import pandas as pd
import torch
import numpy as np
import faiss
from flytekit import task, Resources
from unidecode import unidecode
from deep_translator import GoogleTranslator


from destinations_similarity.scraper.extractor import WikiExtractor
from destinations_similarity.scraper.brazilian_cities import (
    get_brazilian_cities_data, get_dataframe
)
from destinations_similarity.preprocessing_data.text_processing import translate_description_series
from destinations_similarity.preprocessing_data.text_processing import preprocess_text, translate_description
from destinations_similarity.feature_engineering.text_feature_engineering import TextVectorizer


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

@task(retries=3, requests=BASE_RESOURCES)
def preprocess_input_data(dataframe:pd.DataFrame, columns_to_translate: List[str],
                            columns_to_process:List[str]) -> pd.DataFrame:
  for col in columns_to_process+columns_to_translate:
      if col in columns_to_translate:
          dataframe[col] = translate_description_series(dataframe,col)
      else:
          dataframe[col] = dataframe[col].fillna("").swifter.apply(lambda x: unidecode(x) if type(x)==str else x).str.lower()
          dataframe[col] = preprocess_text(dataframe,col)
  return dataframe

@task(retries=3, requests=BASE_RESOURCES)
def vectorize_columns(dataframe,columns_to_vec,city_column) -> List[pd.DataFrame]:
  model = TextVectorizer()
  col_emb=[]
  for col in columns_to_vec:
    inputs_ids = model.encode_inputs(dataframe[col])
    embeddings = model.get_df_embedding(inputs_ids)
    city_embeddings = pd.concat([dataframe[city_column],embeddings],axis=1)
    col_emb.append(city_embeddings)
  return col_emb

@task(retries=3, requests=BASE_RESOURCES)
def build_mean_embedding(list_dataframes:List[pd.DataFrame]) -> torch.tensor:
  col_emb_=[data.iloc[:,1:].values for data in list_dataframes]
  aux = torch.tensor(col_emb_)
  aux_mean = aux.mean(axis=0)
  aux_mean = pd.DataFrame(aux_mean).astype("float")
  aux_mean['city'] = list_dataframes[0].iloc[:,0]
  aux_mean = aux_mean.fillna(0)
  return aux_mean

@task(retries=3, requests=BASE_RESOURCES)
def get_k_most_near(dataframe, kneighborhood: int, vector_dim: int,
                    city_name:str,city_column: str)->None:
        """
        Method responsable for getting the nearst city to a given hotel
        Args:
            knn (int): number of nea hotels
            dim (int): embedding dimension
            city_name (str): city name of hotels
        """
        vec = dataframe[dataframe[city_column]!=city_name].iloc[:,:vector_dim]
        vec_name = dataframe[dataframe[city_column]!=city_name]
        index = faiss.IndexFlatL2(vector_dim)
        index.add(np.ascontiguousarray(np.float32(vec.values)))
        
        query = dataframe[dataframe[city_column]==city_name].iloc[:,:vector_dim].values

        query = np.float32(query)
        
        D, I = index.search(query, kneighborhood)
      
        most_near = vec_name[city_column].iloc[I[0]]
        
        return most_near

@task(retries=3, requests=BASE_RESOURCES)
def build_output(dataframe,nearst_city,see_wikivoyage,
                    do_wikivoyage,city_column,kneighborhood,
                    actual_city_name) -> dict:

    pois_target = dataframe[dataframe[city_column]==actual_city_name][[see_wikivoyage, do_wikivoyage]]
    
    if len(pois_target)>0:
        to_see_actual = translate_description(pois_target[see_wikivoyage].iloc[0],'en')
        to_do_actual = translate_description(pois_target[do_wikivoyage].iloc[0],'en')
    else:
        to_see_actual = "\nOops..unfortunately we don't have the record of what Kin did in this city :/\n"
        to_do_actual = "\nOops..unfortunately we don't have the record of what Kin did in this city :/\n"
    
    
    to_see_nearst = []
    to_do_nearst = []
    for cits in range(kneighborhood):
        pois_suggestion = dataframe[dataframe[city_column]==nearst_city.iloc[cits]][[see_wikivoyage, do_wikivoyage]]
        if len(pois_suggestion)>0:
            to_see_nearst.append(translate_description(pois_suggestion[see_wikivoyage].iloc[0],'en'))
            to_do_nearst.append(translate_description(pois_suggestion[do_wikivoyage].iloc[0],'en'))
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