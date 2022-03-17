# coding=utf-8
"""
Main module created by scrapper
"""
import json

from src.source.brazilian_cities import get_brazilian_cities_data, get_dataframe
from src.drivers.wikipedia import WikipediaExtractor
from src.drivers.wikivoyage import WikivoyagerExtractor
import pandas as pd


def main():
    df = get_brazilian_cities_data(get_dataframe, generate_city_id=True)
    df["search_names"] = df.nome.apply(lambda x: x.replace(" ", "_"))
    voyager_extractor = WikivoyagerExtractor()
    wipedia_extractor = WikipediaExtractor()
    all = []
    for row in range(df.shape[0]):
        data = voyager_extractor.extract_by_name(df.iloc[row].nome, df.iloc[row].search_names)
        if data is None:
            data = wipedia_extractor.extract_by_name(df.iloc[row].nome, df.iloc[row].search_names)
        all.append(data)
    df_cities = pd.DataFrame(all)
    df_cities.to_csv("./sample.csv", index=False, sep=",")


if __name__ == "__main__":
    main()
