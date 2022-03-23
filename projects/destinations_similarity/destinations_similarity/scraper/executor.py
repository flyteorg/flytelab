# coding=utf-8
"""Main module created by scraper."""

from destinations_similarity.scraper.source.brazilian_cities import (
    get_brazilian_cities_data, get_dataframe)
from destinations_similarity.scraper.drivers.wikipedia import WikipediaExtractor
from destinations_similarity.scraper.drivers.wikivoyage import WikivoyageExtractor
import pandas as pd
from tqdm import tqdm


def main():
    df = get_brazilian_cities_data(get_dataframe, generate_city_id=True)
    df["search_names"] = df.nome.apply(lambda x: x.replace(" ", "_"))
    voyage_extractor = WikivoyageExtractor(cities_names=df.nome.values)
    wikipedia_extractor = WikipediaExtractor()
    information_extracted = []
    for row in tqdm(range(df.shape[0])):
        data_voyage = voyage_extractor.extract_by_name(df.iloc[row].nome, df.iloc[row].search_names)
        data_wikipedia = wikipedia_extractor.extract_by_name(df.iloc[row].nome, df.iloc[row].search_names)
        information_extracted.append({**data_voyage, **data_wikipedia})
    df_cities = pd.DataFrame(information_extracted)
    df_cities.to_parquet("./sample2.csv", index=False)
    # normalize related cities
    related_cities = []
    for element in information_extracted:
        if element.get("next") is not None:
            for value in element.get("next"):
                related_cities.append({"city_name": element.get("name"), "next_city_name": value})
    df_next = pd.DataFrame(related_cities)
    df_next.to_csv("./next_city_sample.csv", index=False, sep=",")
    # normalize poi
    city_poi = []
    for element in information_extracted:
        if element.get("poi") is not None:
            for value in element.get("poi"):
                city_poi.append({"city_name": element.get("name"), "poi": value})
    df_poi = pd.DataFrame(city_poi)
    df_poi.to_csv("./poi_city_sample.csv", index=False, sep=",")


if __name__ == "__main__":
    main()
