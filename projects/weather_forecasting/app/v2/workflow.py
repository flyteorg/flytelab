import json
import logging
import os
import time
from io import StringIO
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import List, Optional

import geopy
import pandas as pd
import pandera as pa
import requests
from dataclasses_json import dataclass_json, config
from flytekit import dynamic, task, workflow
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim


USER_AGENT = "flyte-weather-forecasting"
API_BASE_URL = "https://www.ncei.noaa.gov/access/services/search/v1"
DATASET_ENDPOINT = f"{API_BASE_URL}/datasets"
DATA_ENDPOINT = f"{API_BASE_URL}/data"
DATA_ACCESS_URL = "https://www.ncei.noaa.gov"
DATASET_ID = "global-hourly"
MISSING_DATA_INDICATOR = 9999
MAX_RETRIES = 10

logger = logging.getLogger(__file__)

geolocator = Nominatim(user_agent=USER_AGENT)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


class GlobalHourlyDataRaw(pa.SchemaModel):
    date: pa.typing.Series[pa.typing.DateTime]
    tmp: pa.typing.Series[str]

    class Config:
        coerce = True


class GlobalHourlyDataClean(pa.SchemaModel):
    date: pa.typing.Series[pa.typing.DateTime]
    air_temp: pa.typing.Series[float]

    class Config:
        coerce = True


@dataclass_json
@dataclass
class TrainingInstance:
    target_datetime: datetime = field(
        metadata=config(
            encoder=lambda x: datetime.datetime.isoformat(x),
            decoder=lambda x: datetime.datetime.fromisoformat(x),
        )
    )
    features: List[float] = field(
        metadata=config(
            encoder=lambda x: json.dumps([float(i) for i in x]),
            decoder=lambda x: json.loads(x),
        )
    )
    target: Optional[float] = None

    def __post_init__(self):
        if pd.isna(self.target):
            self.target = None


def _get_api_key():
    noaa_api_key = os.getenv("NOAA_API_KEY")
    if noaa_api_key is None:
        raise ValueError("NOAA_API_KEY is not set. Please run `export NOAA_API_KEY=<api_key>`")
    return noaa_api_key


def call_noaa_api(**params):
    params = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    url = f"{DATA_ENDPOINT}?{params}" if params else DATA_ENDPOINT
    logger.debug(f"getting data for request {url}")
    r = requests.get(url, headers={"token": _get_api_key()})
    if r.status_code != 200:
        raise RuntimeError(f"call {url} failed with status code {r.status_code}")
    return r.json()


def bounding_box(location: geopy.Location) -> List[str]:
    """
    Format bounding box

    - from geocoder format: https://nominatim.org/release-docs/develop/api/Output/#boundingbox
    - to NOAA API format: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
    """
    south, north, west, east = range(4)
    return [location.raw["boundingbox"][i] for i in [north, west, south, east]]


def get_global_hourly_data_responses(
    location_query: str, start: datetime, end: datetime = None
) -> List[dict]:
    """Get global hourly data at specified location between two dates."""
    location = geocode(location_query)

    if end is None:
        end = start + datetime.timedelta(days=1)

    logger.debug(f"getting global hourly data for query: {location_query} between {start} and {end}")

    responses = []
    metadata = {"count": -1}
    while metadata["count"] < len(responses):
        metadata = call_noaa_api(
            dataset=DATASET_ID,
            bbox=",".join(bounding_box(location)),
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            units="metric",
            format="json",
            limit=1000,
            offset=len(responses)
        )
        responses.extend(metadata["results"])
    return responses


@pa.check_types
def parse_global_hourly_data(
    df: Optional[pa.typing.DataFrame[GlobalHourlyDataRaw]],
    start: datetime,
    end: datetime,
):
    """Process raw global hourly data.

    For reference, see data documents:
    - https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
    - https://www.ncei.noaa.gov/data/global-hourly/doc/CSV_HELP.pdf
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["air_temp", "air_temp_quality_code", "date"])
    return (
        df["tmp"].str.split(",", expand=True)
        .rename(columns=lambda x: ["air_temp", "air_temp_quality_code"][x])
        .astype(int)
        .query(f"air_temp != {MISSING_DATA_INDICATOR}")
        .join(df["date"])
        .assign(
            # round down to the hour
            date=lambda _: _.date.dt.floor("H"),
            # air temperature is in degress Celsius, with scaling factor of 10
            air_temp=lambda _: _.air_temp * 0.1,
        )
        # remove erroneous data, for details see:
        # page 10, "AIR-TEMPERATURE-OBSERVATION" of
        # https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
        .loc[lambda df: ~df.air_temp_quality_code.isin({3, 7})]
        # only keep records within the time period of interest
        .loc[lambda df: df.date.between(pd.Timestamp(start), pd.Timestamp(end))]
        # get hourly mean air temperature across readings of different sensor locations
        .groupby("date").agg({"air_temp": "mean"})
    )


@task
def get_data_file(filepath: str) -> pd.DataFrame:
    if filepath.startswith("/"):
        filepath = filepath[1:]
        response = requests.get(f"{DATA_ACCESS_URL}/{filepath}")
        return pd.read_csv(StringIO(response.text), low_memory=False)
    raise RuntimeError(f"could not get data file {filepath} from {DATA_ACCESS_URL}")


@task
def process_global_hourly_data(data: List[pd.DataFrame], start: datetime, end: datetime) -> TrainingInstance:
    if len(data) == 0:
        logger.debug(f"no data found")
        return TrainingInstance(target_datetime=end, features=[], target=None)

    hourly_data = (
        pd.concat(data)
        .rename(columns=lambda x: x.lower()).astype({"date": "datetime64[ns]"})
        .pipe(parse_global_hourly_data, start, end)
    )
    target = hourly_data["air_temp"].loc[pd.Timestamp(end)]
    # make sure features are in descending order (most recent first)
    features = hourly_data.loc[
        pd.Timestamp(start): pd.Timestamp(end) - pd.Timedelta(1, unit="H")
    ].sort_index(ascending=False)
    return TrainingInstance(target_datetime=end, features=features["air_temp"].tolist(), target=target)


@dynamic
def get_training_instance(
    location_query: str,
    target_datetime: datetime,
    lookback_window: int,  # one week by default
) -> TrainingInstance:
    target_datetime = target_datetime.replace(minute=0, second=0, microsecond=0)
    start_datetime = target_datetime - timedelta(hours=lookback_window)
    logger.info(
        f"getting global hourly data for query: {location_query} between {start_datetime} and {target_datetime}"
    )
    # get paths to datafiles at dynamic workflow compile time
    responses = get_global_hourly_data_responses(location_query, start_datetime, target_datetime)
    data = []
    logger.debug(f"found {len(responses)} responses")

    for response in responses:
        for station in response["stations"]:
            logger.debug(f"station: {station['name']}")
        data.append(get_data_file(filepath=response['filePath']))

    return process_global_hourly_data(data=data, start=start_datetime, end=target_datetime)


@task
def update_model():
    pass


@dynamic
def get_model():
    pass


@dynamic
def get_forecast():
    pass


@workflow
def forecast_weather(location_query: str, target_datetime: datetime, lookback_window: int) -> TrainingInstance:
    return get_training_instance(
        location_query="Atlanta, GA US",
        target_datetime=datetime.now() - timedelta(days=7),
        lookback_window=24 * 7,
    )


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    hourly_data = get_training_instance(
        location_query="Atlanta, GA US",
        target_datetime=datetime.now() - timedelta(days=7),
        lookback_window=24 * 7,
    )
    import ipdb; ipdb.set_trace()
