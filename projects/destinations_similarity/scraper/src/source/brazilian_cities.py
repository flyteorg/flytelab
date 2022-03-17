# coding=utf-8
"""
--
"""
from requests import get
import pandas as pd
import io
from typing import Any

PUBLIC_GIST_URL = "https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv"


def get_dataframe(file_object: object, *args, **kwargs) -> pd.DataFrame:
    """
    Args:
    Returns:
    """
    if not isinstance(file_object, io.IOBase):
        raise TypeError("Parameter 'file_object' need to be a file-like object")
    dataframe = pd.read_csv(file_object, sep=",")
    if kwargs.get("generate_city_id"):
        dataframe["city_id"] = [(row + 1) for row in range(dataframe.shape[0])]
    return dataframe


def save_csv_data(file_object: object, *args, **kwargs):
    """
    Args:
    Returns:
    """
    if not isinstance(file_object, io.IOBase):
        raise TypeError("Parameter 'file_object' need to be a file-like object")
    dataframe = pd.read_csv(file_object, sep=",")
    if kwargs.get("generate_city_id"):
        dataframe["city_id"] = [(row + 1) for row in range(dataframe.shape[0])]
    if kwargs.get("path"):
        dataframe.to_csv(kwargs.get("path"), index=False)
    else:
        dataframe.to_csv("./brazilian_cities_data.csv", index=False)


def get_brazilian_cities_data(save_data: callable = save_csv_data, *args, **kwargs) -> Any:
    """
    Args:
    Returns:
    """
    request = get(url="https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv")
    if request.status_code == 200:
        buffer = io.BytesIO()
        buffer.write(request.content)
        buffer.seek(0)
        return save_data(buffer, *args, **kwargs)
    else:
        raise ValueError("Unexpected status code returned: %s" % request.status_code)
