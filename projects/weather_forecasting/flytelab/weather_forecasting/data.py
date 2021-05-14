import datetime
import logging
import os
import pandas as pd
import pandera as pa
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from functools import partial, lru_cache
from io import StringIO
from typing import List, Optional

import geopy
import requests
import pytz
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

from flytelab.weather_forecasting import types


USER_AGENT = "flyte-weather-forecasting"
API_BASE_URL = "https://www.ncei.noaa.gov/access/services/search/v1"
DATASET_ENDPOINT = f"{API_BASE_URL}/datasets"
DATA_ENDPOINT = f"{API_BASE_URL}/data"
DATA_ACCESS_URL = "https://www.ncei.noaa.gov"
DATASET_ID = "global-hourly"
MISSING_DATA_INDICATOR = 9999


logger = logging.getLogger(__file__)
geolocator = Nominatim(user_agent=USER_AGENT)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
tf = TimezoneFinder()


class GlobalHourlyData(pa.SchemaModel):
    date: pa.typing.Series[pa.typing.DateTime]

    class Config:
        coerce = True


@dataclass_json
@dataclass
class RawTrainingInstance:
    target_data: pd.DataFrame
    past_days_data: pd.DataFrame
    past_years_data: pd.DataFrame


@dataclass_json
@dataclass
class TrainingInstance:
    features: List[float]
    target: Optional[float]
    target_is_complete: bool  # whether or not target is based on 24 hours worth of weather data
    target_date: types.DateType = field(metadata=types.date_field_config())
    id: Optional[str] = None

    def __post_init__(self):
        if pd.isna(self.target):
            self.target = None


@dataclass_json
@dataclass
class Batch:
    training_data: List[TrainingInstance]
    validation_data: List[TrainingInstance]


def _get_api_key():
    noaa_api_key = os.getenv("NOAA_API_KEY")
    if noaa_api_key is None:
        raise ValueError("NOAA_API_KEY is not set. Please run `export NOAA_API_KEY=<api_key>`")
    return noaa_api_key


def call_noaa_api(url, **params):
    params = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    if params:
        url = f"{url}?{params}"
    logger.debug(f"getting data for request {url}")
    r = requests.get(url, headers={"token": _get_api_key()})
    if r.status_code != 200:
        raise RuntimeError(f"call {url} failed with status code {r.status_code}")
    return r.json()


@lru_cache
def get_location(location_query: str) -> geopy.Location:
    return geocode(location_query)


def bounding_box(location: geopy.Location) -> List[str]:
    """
    Format bounding box

    - from geocoder format: https://nominatim.org/release-docs/develop/api/Output/#boundingbox
    - to NOAA API format: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
    """
    south, north, west, east = range(4)
    return [location.raw["boundingbox"][i] for i in [north, west, south, east]]


def get_timezone(location: geopy.Location):
    return pytz.timezone(tf.timezone_at(lng=location.longitude, lat=location.latitude))


def date_n_years_ago(date: types.DateType, n: int):
    try:
        return date.replace(year=date.year - n)
    except ValueError:
        assert date.month == 2 and date.day == 29  # handle leap year case for 2/29
        return date.replace(day=28, year=date.year - n)


def get_global_hourly_data(location_query: str, start_date: types.DateType, end_date: Optional[types.DateType] = None):
    """Get global hourly data at specified location between two dates."""
    location = get_location(location_query)

    if end_date is None:
        end_date = start_date + datetime.timedelta(days=1)

    logger.debug(f"getting global hourly data for query: {location_query} between {start_date} and {end_date}")

    def get_data(offset):
        return call_noaa_api(
            DATA_ENDPOINT,
            dataset=DATASET_ID,
            bbox=",".join(bounding_box(location)),
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
            units="metric",
            format="json",
            limit=1000,
            offset=offset,
        )

    results = []
    metadata = {"count": -1}
    while metadata["count"] < len(results):
        metadata = get_data(offset=len(results))
        results.extend(metadata["results"])

    data = []
    logger.debug(f"found {len(results)} results")
    for result in results:
        for station in result["stations"]:
            logger.debug(f"station: {station['name']}")
        response = requests.get(f"{DATA_ACCESS_URL}/{result['filePath']}")
        data.append(pd.read_csv(StringIO(response.text), low_memory=False))

    if len(data) == 0:
        logger.debug(f"no data found between {start_date} and {end_date} for query: {location_query}")
        return None

    data = GlobalHourlyData.validate(
        pd.concat(data).rename(columns=lambda x: x.lower())
    )
    return data[data.date.between(pd.Timestamp(start_date), pd.Timestamp(end_date))]


@pa.check_types
def parse_global_hourly_data(df: Optional[pa.typing.DataFrame[GlobalHourlyData]]):
    """Process raw global hourly data.

    For reference, see data document: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["air_temp", "air_temp_quality_code", "date"])
    return (
        df.tmp.str.split(",", expand=True)
        .rename(columns=lambda x: ["air_temp", "air_temp_quality_code"][x])
        .astype(int)
        .query(f"air_temp != {MISSING_DATA_INDICATOR}")
        .join(df["date"])
        .assign(
            date=lambda _: _.date.dt.date,
            # air temperature is in degress Celsius, with scaling factor of 10
            air_temp=lambda _: _.air_temp * 0.1,
        )
    )


def target_is_complete(target_data: pa.typing.DataFrame[GlobalHourlyData]) -> bool:
    """Return True if the target data has a full days worth of data for all stations."""
    return bool(
        target_data
        .groupby("station")["date"]
        .agg(["min", "max"])
        .pipe(lambda df: df["max"] - df["min"])
        # there's some variation in how often data is collected depending on the station.
        # The assumption is that if the data for the target date spans at least 23 hours
        # then consider the sample as complete.
        .pipe(lambda s: (s.dt.seconds / 60 / 60) >= 23)
        .all()
    )


def process_raw_training_instance(raw_training_instance: RawTrainingInstance):
    return (
        pd.concat([
            parse_global_hourly_data(raw_training_instance.target_data),
            parse_global_hourly_data(raw_training_instance.past_days_data),
            parse_global_hourly_data(raw_training_instance.past_years_data),
        ])
        .groupby("date").air_temp.agg("mean").rename("air_temp_mean")
        .sort_index(ascending=False)
    )


def get_training_instance(
    location_query: str,
    target_date: types.DateType,
    lookback_window: int = 14,
    n_year_lookback: int = 1,
    instance_id: str = None,
) -> TrainingInstance:
    """Get a single training instance.

    A single training instance for this model is defined as a `feature`, `target` pair:
    - `target`: the mean temperature on a particular target date `t`
    - `features`:
      - mean temperature from the past `lookback_window` days.
      - mean temperature on target date `t` from the past `n_year_lookback` years.
    """
    get_data = partial(get_global_hourly_data, location_query)
    target_data = get_data(start_date=target_date)
    past_days_data = get_data(
        start_date=target_date - datetime.timedelta(days=lookback_window),
        end_date=target_date,
    )
    past_days_dates = [target_date - datetime.timedelta(days=i) for i in range(1, lookback_window + 1)]

    past_years_data = []
    past_years_dates = []
    for i in range(n_year_lookback):
        past_year_date = date_n_years_ago(target_date, i + 1)
        past_years_dates.append(past_year_date)
        past_years_data.append(
            get_data(start_date=past_year_date).loc[lambda _: _.date.dt.date == past_year_date]
        )

    past_years_data = pd.concat(past_years_data)
    training_instance = process_raw_training_instance(
        RawTrainingInstance(target_data, past_days_data, past_years_data)
    ).reindex([target_date] + past_days_dates + past_years_dates)

    target: Optional[float] = training_instance.get(target_date)
    features = training_instance[training_instance.index < target_date]

    assert features.index.is_monotonic_decreasing, "feature index (by date) should be monotonically decreasing"
    n_expected_features = lookback_window + n_year_lookback
    assert features.shape[0] == n_expected_features, \
        f"expected {n_expected_features} features, found {features.shape[0]}"

    return TrainingInstance(
        features.tolist(),
        target,
        target_is_complete=False if target_data is None else target_is_complete(target_data),
        target_date=target_date,
        id=instance_id,
    )


if __name__ == "__main__":
    target_date = datetime.datetime.now() - datetime.timedelta(days=3)
    training_instance = get_training_instance("Atlanta, GA US", target_date.date())
