# coding=utf-8
"""Module used to extract the base data from the wikis."""

import json
from typing import Any

import requests
import pandas as pd


WIKIDATA_ENDPOINT = 'https://query.wikidata.org/sparql'

WIKIDATA_QUERY = """
    PREFIX schema: <http://schema.org/>

    SELECT ?cityLabel ?stateLabel ?wikivoyageLabel ?wikipediaLabel WHERE {
        ?city wdt:P31 wd:Q3184121;
              wdt:P131 ?state.
        OPTIONAL {
            ?wikipedia schema:about ?city.
            ?wikipedia schema:isPartOf <https://pt.wikipedia.org/>;
                    schema:name ?wikipediaLabel.
        }
        OPTIONAL {
            ?wikivoyage schema:about ?city.
            ?wikivoyage schema:isPartOf <https://en.wikivoyage.org/>;
                        schema:name ?wikivoyageLabel.
        }
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
        }
    }
"""


def get_dataframe(df_object: object, **kwargs) -> pd.DataFrame:
    """Generate a pandas DataFrame from a DataFrame-like object.

    Args:
        df_object (object): A DataFrame-like object (dict, list, etc).

    Returns:
        pd.DataFrame: The DataFrame.
    """
    dataframe = pd.DataFrame(df_object)

    if kwargs.get('generate_city_id'):
        dataframe['city_id'] = [(row + 1) for row in range(dataframe.shape[0])]

    return dataframe


def get_brazilian_cities_data(save_data: callable, *args, **kwargs) -> Any:
    """Get data from brazilian cities from Wikimedia pages.

    Args:
        save_data (callable): Function to process the retrieved data.

    Returns:
        Any: Type returned by save_data.
    """
    request = requests.get(
        WIKIDATA_ENDPOINT,
        params={
            'query': WIKIDATA_QUERY,
            'format': 'json',
        },
        allow_redirects=True,
        stream=True,
    )

    response = json.loads(request.text)
    cities_raw = response['results']['bindings']

    cities = sorted([{
        'city': elem.get('cityLabel', {}).get('value'),
        'state': elem.get('stateLabel', {}).get('value'),
        'title_wikipedia_pt': elem.get('wikipediaLabel', {}).get('value'),
        'title_wikivoyage_en': elem.get('wikivoyageLabel', {}).get('value'),
    } for elem in cities_raw], key=lambda x: x['city'])

    return save_data(cities, *args, **kwargs)
