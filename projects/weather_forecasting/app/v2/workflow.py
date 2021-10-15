import math
import logging
import os
from re import I
import time
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, NamedTuple, Optional

import geopy
import joblib
import numpy as np
import pandas as pd
import pandera as pa
import requests
from dataclasses_json import dataclass_json
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import MultiOutputRegressor

from flytekit import dynamic, task, workflow, CronSchedule, LaunchPlan, Resources, Slack
from flytekit.models.core.execution import WorkflowExecutionPhase
from flytekit.types.file import JoblibSerializedFile


USER_AGENT = "flyte-weather-forecasting"
API_BASE_URL = "https://www.ncei.noaa.gov/access/services/search/v1"
DATASET_ENDPOINT = f"{API_BASE_URL}/datasets"
DATA_ENDPOINT = f"{API_BASE_URL}/data"
DATA_ACCESS_URL = "https://www.ncei.noaa.gov"
DATASET_ID = "global-hourly"
MISSING_DATA_INDICATOR = 9999
MAX_RETRIES = 10
CACHE_VERSION = "1.8"

logger = logging.getLogger(__file__)

request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")

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
class Features:
    air_temp_features: List[float]
    dew_temp_features: List[float]
    time_based_feature: Optional[datetime]

    def __post_init__(self):
        if isinstance(self.time_based_feature, float):
            self.time_based_feature = datetime.fromtimestamp(self.time_based_feature)


@dataclass_json
@dataclass
class Target:
    air_temp: float
    dew_temp: float

    def __post_init__(self):
        if self.air_temp == "NaN":
            self.air_temp = float("nan")
        if self.dew_temp == "NaN":
            self.dew_temp = float("nan")


@dataclass_json
@dataclass
class TrainingInstance:
    target_datetime: Optional[datetime]
    features: Features
    target: Target

    def __post_init__(self):
        air_temp = float("nan") if pd.isna(self.target.air_temp) else self.target.air_temp
        dew_temp = float("nan") if pd.isna(self.target.dew_temp) else self.target.dew_temp
        self.target = Target(air_temp, dew_temp)


@dataclass_json
@dataclass
class Scores:
    # keep track of mean absolute error
    train_exp_mae: float = 0.0
    valid_exp_mae: float = 0.0


@dataclass_json
@dataclass
class Prediction:
    air_temp: Optional[float]
    dew_temp: Optional[float]
    date: datetime
    error: Optional[str] = None
    imputed: bool = False


@dataclass_json
@dataclass
class Forecast:
    created_at: datetime
    model_id: str
    predictions: List[Prediction]


ApiResult = NamedTuple("ApiResult", results=List[dict], count=int)
ModelUpdate = NamedTuple("ModelUpdate", model_file=JoblibSerializedFile, scores=Scores)
LatestModelBundle = NamedTuple(
    "ModelUpdate", model_file=JoblibSerializedFile, scores=Scores, training_instance=TrainingInstance
)
WeatherForecast = NamedTuple("WeatherForecast", forecast=Forecast, scores=Scores)


EMPTY_TRAINING_INSTANCE = TrainingInstance(
    target_datetime=None,
    features=Features(
        air_temp_features=[],
        dew_temp_features=[],
        time_based_feature=None,
    ),
    target=Target(float("nan"), float("nan"))
)


def _get_api_key():
    noaa_api_key = os.getenv("NOAA_API_KEY")
    if noaa_api_key is None:
        raise ValueError("NOAA_API_KEY is not set. Please run `export NOAA_API_KEY=<api_key>`")
    return noaa_api_key


def call_noaa_api(
    bounding_box: str,
    start: datetime,
    end: datetime,
    responses: List[List[dict]],
) -> ApiResult:
    """Call the NOAA API to get data.

    See https://www.ncdc.noaa.gov/cdo-web/webservices/v2 for request limits.
    """
    params = dict(
        dataset=DATASET_ID,
        bbox=bounding_box,
        startDate=start.replace(tzinfo=None).isoformat(),
        endDate=end.replace(tzinfo=None).isoformat(),
        units="metric",
        format="json",
        limit=1000,
        offset=len([x for results in responses for x in results]),
    )
    params = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    url = f"{DATA_ENDPOINT}?{params}" if params else DATA_ENDPOINT
    logger.debug(f"getting data for request {url}")
    r = requests.get(url, headers={"token": _get_api_key()})
    time.sleep(0.25)  # limit to four requests per second
    if r.status_code != 200:
        raise RuntimeError(f"call {url} failed with status code {r.status_code}")
    metadata = r.json()
    return ApiResult(results=metadata["results"], count=metadata["count"])


def bounding_box(location: geopy.Location) -> List[str]:
    """
    Format bounding box

    - from geocoder format: https://nominatim.org/release-docs/develop/api/Output/#boundingbox
    - to NOAA API format: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
    """
    south, north, west, east = range(4)
    return [location.raw["boundingbox"][i] for i in [north, west, south, east]]


def get_global_hourly_data_responses(
    location_query: str, start: datetime, end: datetime
) -> List[dict]:
    """Get global hourly data at specified location between two dates."""
    start = start.replace(tzinfo=None)
    end = end.replace(tzinfo=None)
    logger.debug(f"getting global hourly data for query: {location_query} between {start} and {end}")
    location = geocode(location_query)
    responses = []
    count = -1
    while count < len([x for results in responses for x in results]):
        results, count = call_noaa_api(
            bounding_box=",".join(bounding_box(location)),
            start=start,
            end=end,
            responses=responses,
        )
        responses.append(results)
    return [x for results in responses for x in results]


def parse_temperature(temp: pd.Series, suffix: str) -> pd.Series:
    temp_col, quality_code_col = f"{suffix}temp", f"{suffix}temp_quality_code"
    return (
        temp.str.split(",", expand=True)
        .rename(columns={0: temp_col, 1: quality_code_col})
        .astype(int)
        .query(f"{temp_col} != {MISSING_DATA_INDICATOR}")
        .assign(**{
            # air temperature is in degress Celsius, with scaling factor of 10
            temp_col: lambda _: _[temp_col] * 0.1
        })
    )


@pa.check_types
def parse_global_hourly_data(df: Optional[pa.typing.DataFrame[GlobalHourlyDataRaw]]):
    """Process raw global hourly data.

    For reference, see data documents:
    - https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
    - https://www.ncei.noaa.gov/data/global-hourly/doc/CSV_HELP.pdf
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["air_temp", "air_temp_quality_code", "date"])
    out = (
        # "tmp" is a string column and needs to be further parsed
        parse_temperature(df["tmp"], suffix="air_")
        .join(parse_temperature(df["dew"], suffix="dew_"))
        .join(df["date"])
        .assign(
            # round down to the hour
            date=lambda _: _.date.dt.floor("H"),
        )
        # remove erroneous data, for details see:
        # page 10, "AIR-TEMPERATURE-OBSERVATION" of
        # https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
        .loc[lambda df: ~df.air_temp_quality_code.isin({3, 7})]
    )
    return (
        # get hourly mean air temperature across readings of different sensor locations
        out.groupby("date").agg({"air_temp": "mean", "dew_temp": "mean"})
    )


def get_data_file(filepath: str) -> pd.DataFrame:
    """Get raw data file from global hourly archive.

    https://www.ncei.noaa.gov/data/global-hourly

    NOTE: the `start_date` and `end_date` arguments are primarily for caching. The filepath points to a remote csv
    file that is frequently updated, so this task needs to know from the API response whether the `end_date` has
    changed so this task can fetch an updated version of the csv.
    """
    if filepath.startswith("/"):
        filepath = filepath[1:]
        response = requests.get(f"{DATA_ACCESS_URL}/{filepath}")
        time.sleep(0.25)  # limit to four requests per second
        return pd.read_csv(StringIO(response.text), low_memory=False)
    raise RuntimeError(
        f"could not get data file {filepath} from {DATA_ACCESS_URL}"
    )


def get_raw_data(responses: List[dict]) -> List[pd.DataFrame]:
    data = []
    logger.debug(f"found {len(responses)} responses")
    # TODO: figure out how to not cache a data file if doesn't contain data for all hours of the day
    for response in responses:
        for station in response["stations"]:
            print(f"station: {station['name']}")
        data.append(get_data_file(filepath=response['filePath']))
    return data


def process_raw_training_data(data: List[pd.DataFrame]) -> pd.DataFrame:
    if len(data) == 0:
        logger.debug(f"no data found")
        return pd.DataFrame()
    return (
        pd.concat(data)
        .rename(columns=lambda x: x.lower()).astype({"date": "datetime64[ns]"})
        .pipe(parse_global_hourly_data)
    )


@task(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def get_training_data(
    location_query: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    logger.info(
        f"getting global hourly data for query: {location_query} between {start} and {end}"
    )
    responses = get_global_hourly_data_responses(location_query=location_query, start=start, end=end)
    return process_raw_training_data(data=get_raw_data(responses=responses))


@task(
    requests=request_resources,
    limits=limit_resources,
)
def prepare_training_instance(training_data: pd.DataFrame, start: datetime, end: datetime) -> TrainingInstance:
    # make sure features are in descending order (most recent first)
    start = start.replace(tzinfo=None)
    end = end.replace(tzinfo=None)

    features = training_data.loc[
        pd.to_datetime(start): pd.to_datetime(end) - pd.Timedelta(1, unit="H")
    ].sort_index(ascending=False)

    if training_data.empty or features.empty:
        return TrainingInstance(
            target_datetime=end.replace(tzinfo=None),
            features=Features(
                air_temp_features=[],
                dew_temp_features=[],
                time_based_feature=None,
            ),
            target=Target(float("nan"), float("nan"))
        )

    if pd.to_datetime(end) in training_data.index:
        target = Target(
            air_temp=training_data["air_temp"].loc[pd.to_datetime(end)],
            dew_temp=training_data["dew_temp"].loc[pd.to_datetime(end)],
        )
    else:
        target = Target(float("nan"), float("nan"))
    return TrainingInstance(
        target_datetime=end.replace(tzinfo=None),
        features=Features(
            air_temp_features=features["air_temp"].tolist(),
            dew_temp_features=features["dew_temp"].tolist(),
            time_based_feature=features.index[0].to_pydatetime(),
        ),
        target=target
    )


def round_datetime_to_year(dt: datetime, ceil: bool = True) -> datetime:
    """Round date-times to the year."""
    return datetime(dt.year + 1 if ceil else dt.year + 1, 1, 1, 0, tzinfo=None)


@dynamic(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def get_training_instance(location_query: str, start: datetime, end: datetime) -> TrainingInstance:
    training_data = get_training_data(
        location_query=location_query,
        start=round_datetime_to_year(dt=start, ceil=False),
        end=round_datetime_to_year(dt=end, ceil=True),
    )
    return prepare_training_instance(training_data=training_data, start=start, end=end)


@task(cache=True, cache_version=CACHE_VERSION)
def init_model(genesis_datetime: datetime) -> JoblibSerializedFile:
    """Initialize the model."""
    base_model = SGDRegressor(
        penalty="l2",
        alpha=0.001,
        random_state=int(genesis_datetime.timestamp()),
        learning_rate="constant",
        eta0=0.1,
        warm_start=True,
        average=True,
    )
    model = MultiOutputRegressor(estimator=base_model)
    model_fp = f"/tmp/model.joblib"
    with open(model_fp, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=model_fp)


@task
def get_previous_datetime(current_datetime: datetime) -> datetime:
    """Update the model hourly."""
    return current_datetime - timedelta(hours=1)


def onehot_encode(x: int, k_values: int):
    return [int(x == i) for i in range(k_values)]


def minmax_scaler(x: int, min_value: int, max_value: int):
    """Scales values with lower and upper bounds to within 0 and 1."""
    return (x - min_value) / (max_value - min_value)


def encode_datetime(dt: datetime):
    """One-hot encode datetime into features."""
    dt = pd.Timestamp(dt)
    return np.array([
        *onehot_encode(dt.hour, 24),
        *onehot_encode(dt.day_of_week, 7),
        *onehot_encode(dt.day, 31),  # day of month
        *onehot_encode(dt.day_of_year, 356),
        *onehot_encode(dt.month, 12),  # month of year
        # represent year as numerical feature between 0 and 1 based on a
        # min and max range of 1900-2500 (arbitrary, I know). The value just
        # needs to be small enough to not throw off training.
        minmax_scaler(dt.year, 1900, 2500)
    ])


def encode_features(features: Features):
    """Mean-center and scale by standard-deviation.

    NOTE: This feature space is huge!
    """
    # only encode the date of the most recent data point
    # TODO: add temperature differences as a feature
    air_temp = minmax_scaler(np.array(features.air_temp_features), -100, 100)
    dew_temp = minmax_scaler(np.array(features.dew_temp_features), -100, 100)

    # difference in temperatures relative to the latest temp
    air_temp_diffs = air_temp[0] - air_temp[1:]
    dew_temp_diffs = dew_temp[0] - dew_temp[1:]

    # time-based feature encoding
    time_features = encode_datetime(features.time_based_feature)

    # interaction term between temp and date
    interaction_features_air_temp = air_temp[0] * time_features
    interaction_features_dew_temp = dew_temp[0] * time_features

    # interaction term between temp diffs and date
    interaction_features_air_diff = np.matmul(air_temp_diffs.reshape(-1, 1), time_features.reshape(1, -1)).ravel()
    interaction_features_dew_diff = np.matmul(dew_temp_diffs.reshape(-1, 1), time_features.reshape(1, -1)).ravel()

    return np.concatenate([
        # min-max scale temperate based on range of -100, 100 degrees celsius
        air_temp[:1],
        dew_temp[:1],
        air_temp_diffs,
        dew_temp_diffs,
        time_features,
        interaction_features_air_temp,
        interaction_features_air_diff,
        interaction_features_dew_temp,
        interaction_features_dew_diff,
    ]).reshape(1, -1)


def exp_weighted_mae(mae, exp_mae, epsilon=0.1):
    return epsilon * mae + (1 - epsilon) * exp_mae


def encode_targets(targets: Target):
    return [[targets.dew_temp, targets.air_temp]]


@task(cache=True, cache_version=CACHE_VERSION)
def update_model(
    model: JoblibSerializedFile,
    training_instance: TrainingInstance,
    scores: Scores,
) -> ModelUpdate:
    logging.info("updating model")
    with open(model, "rb") as f:
        model = joblib.load(f)

    features = encode_features(training_instance.features)

    try:
        y_pred = model.predict(features).ravel()[1]
    except NotFittedError:
        # model can't predict at initialization, so use mean of features as prediction
        y_pred = np.mean(training_instance.features.air_temp_features)

    # compute running validation mean absolute error before updating the model
    valid_exp_mae = exp_weighted_mae(abs(y_pred - training_instance.target.air_temp), scores.valid_exp_mae)

    # update the model
    model.partial_fit(features, encode_targets(training_instance.target))

    # compute running training mean absolute error after the update
    train_exp_mae = exp_weighted_mae(
        abs(model.predict(features).ravel()[1] - training_instance.target.air_temp),
        scores.valid_exp_mae
    )

    model_fp = f"/tmp/model.joblib"
    with open(model_fp, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=model_fp), Scores(train_exp_mae, valid_exp_mae)

@task
def stop_training(training_instance: TrainingInstance) -> bool:
    return math.isnan(training_instance.target.air_temp)


@dynamic(cache=True, cache_version=CACHE_VERSION)
def get_latest_model(
    location_query: str,
    target_datetime: datetime,
    genesis_datetime: datetime,
    lookback_window: int,
) -> LatestModelBundle:
    """Iterate from genesis date to target date to get the most up-to-date model.

    This dynamic workflow iterates from genesis datetime to target datetime to return the
    most up-to-date model. Note that this takes advantage of Flyte caching to ensure that
    the model doesn't have to be re-trained every time it's called; it hits the cache
    until it reaches the target date, which presumably hasn't been called yet.
    """
    target_datetime = target_datetime.replace(tzinfo=None)
    genesis_datetime = genesis_datetime.replace(tzinfo=None)

    model = init_model(genesis_datetime=genesis_datetime)
    scores = Scores()

    diff_in_hours = (target_datetime - genesis_datetime).days * 24
    for i in range(1, diff_in_hours + 1):
        current_datetime = genesis_datetime + timedelta(hours=i)
        training_instance = get_training_instance(
            location_query=location_query,
            start=current_datetime - timedelta(hours=lookback_window),
            end=current_datetime,
        )
        if stop_training(training_instance=training_instance):
            return model, scores, training_instance
        model, scores = update_model(model=model, training_instance=training_instance, scores=scores)
    return model, scores, training_instance


@task
def get_forecast(
    model: JoblibSerializedFile,
    latest_training_instance: TrainingInstance,
    target_datetime: datetime,
    forecast_window: int,
) -> Forecast:
    with open(model, "rb") as f:
        model = joblib.load(f)

    predictions = []
    features = latest_training_instance.features

    target_datetime = target_datetime.replace(tzinfo=None)
    latest_datetime = latest_training_instance.target_datetime.replace(tzinfo=None)
    diff = (target_datetime - latest_datetime).days * 24

    n_forecasts = diff + forecast_window

    latest_datetime = latest_training_instance.target_datetime.replace(tzinfo=None)
    for i in range(n_forecasts + 1):
        try:
            prediction = model.predict(encode_features(features)).ravel()
            error = None
        except Exception as e:
            prediction = None
            error = str(e)

        # make hourly predictions
        current_datetime = latest_datetime + timedelta(hours=i)
        predictions.append(
            Prediction(
                air_temp=prediction[1],
                dew_temp=prediction[0],
                date=current_datetime,
                error=error,
                # keep track of whether a prediction is imputed because the training
                # data on a `date <= target_datetime` didn't contain  target labels.
                imputed=current_datetime <= target_datetime,
            )
        )
        features = Features(
            air_temp_features=[prediction[1]] + features.air_temp_features[:-1],
            dew_temp_features=[prediction[0]] + features.dew_temp_features[:-1],
            time_based_feature=current_datetime,
        )

    return Forecast(created_at=target_datetime, model_id=joblib.hash(model), predictions=predictions)


@task
def normalize_datetime(dt: datetime) -> datetime:
    """Round date-times to the nearest hour."""
    return datetime(dt.year, dt.month, dt.day, hour=dt.hour, tzinfo=None)


@workflow
def forecast_weather(
    location_query: str,
    target_datetime: datetime,
    genesis_datetime: datetime,
    lookback_window: int,
    forecast_window: int,
) -> WeatherForecast:
    target_datetime = normalize_datetime(dt=target_datetime)
    genesis_datetime = normalize_datetime(dt=genesis_datetime)
    model, scores, latest_training_instance = get_latest_model(
        location_query=location_query,
        target_datetime=target_datetime,
        genesis_datetime=genesis_datetime,
        lookback_window=lookback_window,
    )
    forecast = get_forecast(
        model=model,
        latest_training_instance=latest_training_instance,
        target_datetime=target_datetime,
        forecast_window=forecast_window,
    )
    return forecast, scores


DEFAULT_INPUTS = {
    "target_datetime": datetime(2021, 9, 1),
    "genesis_datetime": datetime(2021, 7, 1),
    "lookback_window": 24 * 3,  # 3-day lookback
    "forecast_window": 24 * 3,  # 3-day forecast
}

SLACK_NOTIFICATION = Slack(
    phases=[
        WorkflowExecutionPhase.SUCCEEDED,
        WorkflowExecutionPhase.TIMED_OUT,
        WorkflowExecutionPhase.FAILED,
    ],
    recipients_email=[
        "flytlab-notifications-aaaad6weta5ic55r7lmejgwzha@unionai.slack.com",
        "niels@union.ai",
    ],
)


atlanta_lp = LaunchPlan.get_or_create(
    workflow=forecast_weather,
    name="atlanta_weather_forecast_v2",
    default_inputs=DEFAULT_INPUTS,
    fixed_inputs={"location_query": "Atlanta, GA USA"},
    schedule=CronSchedule("0 4 * * ? *"),  # EST midnight
    notifications=[SLACK_NOTIFICATION],
)

seattle_lp = LaunchPlan.get_or_create(
    workflow=forecast_weather,
    name="seattle_weather_forecast_v2",
    default_inputs=DEFAULT_INPUTS,
    fixed_inputs={"location_query": "Seattle, WA USA"},
    schedule=CronSchedule("0 7 * * ? *"),  # PST midnight
    notifications=[SLACK_NOTIFICATION],
)

hyderabad_lp = LaunchPlan.get_or_create(
    workflow=forecast_weather,
    name="hyderabad_weather_forecast_v2",
    default_inputs=DEFAULT_INPUTS,
    fixed_inputs={"location_query": "Hyderabad, Telangana India"},
    schedule=CronSchedule("30 18 * * ? *"),  # IST midnight
    notifications=[SLACK_NOTIFICATION],
)



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    forecast, scores = forecast_weather(
        location_query="Atlanta, GA US",
        target_datetime=datetime.now() + timedelta(days=7),
        genesis_datetime=datetime.now() - timedelta(days=7 * 1),
        lookback_window=24 * 1,
        forecast_window=24 * 1,
    )
    print("Forecasts:")
    for prediction in forecast.predictions:
        print(prediction)

    print(f"Scores: {scores}")
