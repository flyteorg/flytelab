import logging
import math
import os
import time
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, NamedTuple, Optional

import joblib
import numpy as np
import pandas as pd
import pandera as pa
import requests
from dataclasses_json import dataclass_json
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from pandera.typing import DataFrame, Series, Index
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import MultiOutputRegressor

from flytekit import dynamic, task, workflow, CronSchedule, LaunchPlan, Resources, Slack, Email
from flytekit.models.core.execution import WorkflowExecutionPhase
from flytekit.types.file import JoblibSerializedFile

import flytekitplugins.pandera  # noqa


USER_AGENT = "flyte-weather-forecasting-agent"
API_BASE_URL = "https://www.ncei.noaa.gov/access/services/search/v1"
DATASET_ENDPOINT = f"{API_BASE_URL}/datasets"
DATA_ENDPOINT = f"{API_BASE_URL}/data"
DATA_ACCESS_URL = "https://www.ncei.noaa.gov"
DATASET_ID = "global-hourly"
MISSING_DATA_INDICATOR = 9999
MAX_RETRIES = 10
CACHE_VERSION = "2.2"

logger = logging.getLogger(__file__)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")

geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=10)


class GlobalHourlyDataRaw(pa.SchemaModel):
    DATE: Series[pa.typing.DateTime]
    TMP: Series[str]

    class Config:
        coerce = True


class GlobalHourlyData(pa.SchemaModel):

    # validate the min and max temperature range in degrees Celsius
    air_temp: Series[float] = pa.Field(ge=-273.15, le=459.67)
    dew_temp: Series[float] = pa.Field(ge=-273.15, le=459.67)
    date: Index[pa.typing.DateTime] = pa.Field(unique=True)

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


@dataclass_json
@dataclass
class BoundingBox:
    north: str
    west: str
    south: str
    east: str


NormalizedDatetimes = NamedTuple("NormalizedDatetimes", genesis_datetime=datetime, target_datetime=datetime)
ModelUpdate = NamedTuple(
    "ModelUpdate",
    model_file=JoblibSerializedFile,
    scores=Scores,
    training_instance=TrainingInstance
)
ApiResult = NamedTuple("ApiResult", results=List[dict], count=int)
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
    bounding_box: BoundingBox,
    start: datetime,
    end: datetime,
    responses: List[List[dict]],
) -> ApiResult:
    """Call the NOAA API to get data.

    See https://www.ncdc.noaa.gov/cdo-web/webservices/v2 for request limits.
    """
    params = dict(
        dataset=DATASET_ID,
        bbox=",".join(bounding_box.to_dict().values()),
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


@task(cache=True, cache_version=CACHE_VERSION)
def get_bounding_box(location_query: str) -> BoundingBox:
    """
    Get geolocation and create bounding box

    - from geocoder format: https://nominatim.org/release-docs/develop/api/Output/#boundingbox
    - to NOAA API format: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
    """
    location = geocode(location_query)
    south, north, west, east = range(4)
    return BoundingBox(*[location.raw["boundingbox"][i] for i in [north, west, south, east]])


def get_global_hourly_data_responses(
    bounding_box: BoundingBox, start: datetime, end: datetime
) -> List[dict]:
    """Get global hourly data at specified location between two dates."""
    start = start.replace(tzinfo=None)
    end = end.replace(tzinfo=None)
    logger.debug(f"getting global hourly data for bounding box: {bounding_box} between {start} and {end}")
    responses = []
    count = -1
    while count < len([x for results in responses for x in results]):
        results, count = call_noaa_api(
            bounding_box=bounding_box,
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
        .astype({temp_col: int})
        .query(f"{temp_col} != {MISSING_DATA_INDICATOR}")
        .assign(**{
            # air temperature is in degress Celsius, with scaling factor of 10
            temp_col: lambda _: _[temp_col] * 0.1
        })
    )


@pa.check_types
def get_data_file(filepath: str) -> DataFrame[GlobalHourlyDataRaw]:
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


def get_raw_data(responses: List[dict]) -> DataFrame[GlobalHourlyDataRaw]:
    data = []
    logger.debug(f"found {len(responses)} responses")
    # TODO: figure out how to not cache a data file if doesn't contain data for all hours of the day
    for response in responses:
        for station in response["stations"]:
            print(f"station: {station['name']}")
        data.append(get_data_file(filepath=response['filePath']))
    return pd.concat(data)


def process_raw_training_data(raw_data: DataFrame[GlobalHourlyDataRaw]) -> DataFrame[GlobalHourlyData]:
    """Process raw global hourly data.

    For reference, see data documents:
    - https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
    - https://www.ncei.noaa.gov/data/global-hourly/doc/CSV_HELP.pdf
    """
    if raw_data.empty:
        return pd.DataFrame(columns=["air_temp", "air_temp_quality_code", "date"])
    raw_data = raw_data.rename(columns=lambda x: x.lower()).astype({"date": "datetime64[ns]"})
    return (
        # "tmp" is a string column and needs to be further parsed
        parse_temperature(raw_data["tmp"], suffix="air_")
        .join(parse_temperature(raw_data["dew"], suffix="dew_"))
        .join(raw_data["date"])
        .assign(
            # round down to the hour
            date=lambda _: _.date.dt.floor("H"),
        )
        # remove erroneous data, for details see:
        # page 10, "AIR-TEMPERATURE-OBSERVATION" of
        # https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
        .loc[lambda df: ~df.air_temp_quality_code.isin({"3", "7"})]
        # get hourly mean air temperature across readings of different sensor locations
        .groupby("date").agg({"air_temp": "mean", "dew_temp": "mean"})
        # resample at hourly interval to make sure all hourly intervals have an entry
        .resample("1H").mean()
        # linearly interpolate into the future
        .interpolate(method="linear", limit_direction="forward", limit=None)
    )


@task(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def get_weather_data(
    bounding_box: BoundingBox,
    start: datetime,
    end: datetime,
    fetch_date: datetime,
) -> DataFrame[GlobalHourlyData]:
    logger.info(
        f"getting global hourly data for query: {bounding_box} between {start} and {end}, fetch date: {fetch_date}"
    )
    return process_raw_training_data(
        raw_data=get_raw_data(
            responses=get_global_hourly_data_responses(bounding_box=bounding_box, start=start, end=end)
        )
    )


def _prepare_training_instance(
    training_data: DataFrame[GlobalHourlyData], start: datetime, end: datetime
) -> TrainingInstance:
    logging.debug(f"get training instance: {start} - {end}")

    # make sure dates are timezone-unaware
    start = pd.to_datetime(start.replace(tzinfo=None))
    end = pd.to_datetime(end.replace(tzinfo=None))

    air_temp_features = []
    dew_temp_features = []
    time_based_feature = None
    target = Target(float("nan"), float("nan"))

    # make sure features are in descending order (most recent first)
    features = training_data.loc[start: end - pd.Timedelta(1, unit="H")].sort_index(ascending=False)

    if not (training_data.empty or features.empty):
        air_temp_features = features["air_temp"].tolist()
        dew_temp_features = features["dew_temp"].tolist()
        time_based_feature = features.index[0].to_pydatetime()
    if pd.to_datetime(end) in training_data.index:
        target = Target(
            air_temp=training_data["air_temp"].loc[end],
            dew_temp=training_data["dew_temp"].loc[end],
        )
    return TrainingInstance(
        target_datetime=end,
        features=Features(
            air_temp_features=air_temp_features,
            dew_temp_features=dew_temp_features,
            time_based_feature=time_based_feature,
        ),
        target=target,
    )


@task(requests=request_resources, limits=limit_resources)
def prepare_training_instance(
    training_data: DataFrame[GlobalHourlyData],
    start: datetime,
    end: datetime,
) -> TrainingInstance:
    return _prepare_training_instance(training_data, start, end)


@task
def round_datetime(dt: datetime, ceil: bool) -> datetime:
    """Round date-times to the nearest hour."""
    return datetime(dt.year + 1 if ceil else dt.year, 1, 1, 0, tzinfo=None)


@task(requests=request_resources, limits=limit_resources)
def instances_from_daterange(
    training_data: DataFrame[GlobalHourlyData],
    start: datetime,
    end: datetime,
    lookback_window: int,
) -> List[TrainingInstance]:
    training_instances = []
    diff_in_hours = int((end - start).total_seconds() / 60 / 60)
    for i in range(1, diff_in_hours + 1):
        current_datetime = start + timedelta(hours=i)
        training_instance = _prepare_training_instance(
            training_data=training_data,
            start=current_datetime - timedelta(hours=lookback_window),
            end=current_datetime,
        )
        training_instances.append(training_instance)
    return training_instances


@task
def datetime_now() -> datetime:
    return pd.Timestamp.now().floor("H").to_pydatetime()


@dynamic(
    requests=request_resources,
    limits=limit_resources,
)
def get_training_instances(
    bounding_box: BoundingBox,
    start: datetime,
    end: datetime,
    lookback_window: int,
) -> List[TrainingInstance]:
    training_data = get_weather_data(
        bounding_box=bounding_box,
        start=round_datetime(dt=start, ceil=False),
        end=round_datetime(dt=end, ceil=True),
        # Make sure the processed weather data cache is invalidated every hour.
        fetch_date=datetime_now(),
    )
    return instances_from_daterange(
        training_data=training_data,
        start=start,
        end=end,
        lookback_window=lookback_window,
    )


def serialize_model(model: BaseEstimator) -> JoblibSerializedFile:
    """Convert model object to compressed byte string."""
    out_file = "/tmp/model.joblib"
    with open(out_file, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=out_file)


def deserialize_model(model_file: JoblibSerializedFile) -> BaseEstimator:
    """Load model object from compressed byte string."""
    with open(model_file, "rb") as f:
        model = joblib.load(f)
    return model


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
    """Mean-center and scale by standard-deviation."""
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


def _update_model(model, scores, training_instance):
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
        scores.train_exp_mae
    )
    logging.debug(f"updated model: train mae={train_exp_mae}, valid mae={valid_exp_mae}")
    return model, Scores(train_exp_mae, valid_exp_mae)


@task
def update_model(
    model: JoblibSerializedFile, scores: Scores, training_instances: List[TrainingInstance]
) -> ModelUpdate:
    model = deserialize_model(model)
    for training_instance in training_instances:
        model, scores = _update_model(model, scores, training_instance)
    return ModelUpdate(model_file=serialize_model(model), scores=scores, training_instance=training_instance)


@dynamic(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def init_model(
    bounding_box: BoundingBox,
    genesis_datetime: datetime,
    n_days_pretraining: int,
    lookback_window: int,
) -> ModelUpdate:
    """Initialize the model."""
    model = MultiOutputRegressor(
        estimator=SGDRegressor(
            penalty="l2",
            alpha=0.001,
            random_state=int(genesis_datetime.timestamp()),
            learning_rate="constant",
            eta0=0.1,
            warm_start=True,
            average=True,
        )
    )
    end = genesis_datetime.replace(tzinfo=None)
    start = genesis_datetime - timedelta(days=n_days_pretraining)
    return update_model(
        model=serialize_model(model),
        scores=Scores(),
        training_instances=get_training_instances(
            bounding_box=bounding_box,
            start=start,
            end=end,
            lookback_window=lookback_window,
        ),
    )


@task
def get_previous_target_datetime(target_datetime: datetime, genesis_datetime: datetime) -> datetime:
    """Get the previous target datetime, rounded down to days since genesis datetime.

    This means that the model will only update every day.
    """
    diff = math.ceil((target_datetime - genesis_datetime).total_seconds() / 60 / 60 / 24)
    prev_datetime = genesis_datetime + timedelta(days=diff) - timedelta(days=1)
    if prev_datetime < genesis_datetime:
        return genesis_datetime
    return prev_datetime


@task
def get_training_instance_datetime(training_instance: TrainingInstance) -> datetime:
    return training_instance.target_datetime


@dynamic(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def get_latest_model(
    bounding_box: BoundingBox,
    target_datetime: datetime,
    genesis_datetime: datetime,
    n_days_pretraining: int,
    lookback_window: int,
) -> ModelUpdate:
    if target_datetime <= genesis_datetime:
        logging.info(f"initializing model at {genesis_datetime}")
        return init_model(
            bounding_box=bounding_box,
            genesis_datetime=genesis_datetime,
            n_days_pretraining=n_days_pretraining,
            lookback_window=lookback_window,
        )
    else:
        previous_target_datetime = get_previous_target_datetime(
            target_datetime=target_datetime,
            genesis_datetime=genesis_datetime,
        )
        prev_model, prev_scores, prev_training_instance = get_latest_model(
            bounding_box=bounding_box,
            target_datetime=previous_target_datetime,
            genesis_datetime=genesis_datetime,
            n_days_pretraining=n_days_pretraining,
            lookback_window=lookback_window,
        )
        logging.info(f"updating model at {target_datetime}")
        logging.info(f"previous update datetime: {previous_target_datetime}")
        return update_model(
            model=prev_model,
            scores=prev_scores,
            training_instances=get_training_instances(
                bounding_box=bounding_box,
                start=get_training_instance_datetime(training_instance=prev_training_instance),
                end=target_datetime,
                lookback_window=lookback_window,
            ),
        )


@task(
    requests=request_resources,
    limits=limit_resources,
)
def get_forecast(
    latest_model: JoblibSerializedFile,
    latest_training_instance: TrainingInstance,
    target_datetime: datetime,
    forecast_window: int,
) -> Forecast:
    model = deserialize_model(latest_model)

    predictions = []
    features = latest_training_instance.features

    target_datetime = target_datetime.replace(tzinfo=None)
    latest_datetime = latest_training_instance.target_datetime.replace(tzinfo=None)
    diff = math.ceil((target_datetime - latest_datetime).total_seconds() / 60 / 60)
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
def round_datetime_to_hour(dt: datetime) -> datetime:
    """Round date-times to the nearest hour."""
    return datetime(dt.year, dt.month, dt.day, hour=dt.hour, tzinfo=None)


@dynamic(
    requests=request_resources,
    limits=limit_resources,
)
def normalize_datetimes(
    genesis_datetime: datetime,
    target_datetime: datetime,
    training_data: DataFrame[GlobalHourlyData],
) -> NormalizedDatetimes:
    """Get the date of the latest available training data."""
    genesis_datetime = genesis_datetime.replace(tzinfo=None)
    target_datetime = target_datetime.replace(tzinfo=None)
    latest_available = training_data.index[-1].to_pydatetime()
    target_datetime = min(target_datetime, latest_available)
    if target_datetime < genesis_datetime:
        genesis_datetime = latest_available
    return genesis_datetime, target_datetime


@workflow
def forecast_weather(
    location_query: str,
    target_datetime: datetime,
    genesis_datetime: datetime,
    n_days_pretraining: int,
    lookback_window: int,
    forecast_window: int,
) -> WeatherForecast:
    bounding_box = get_bounding_box(location_query=location_query)
    genesis_datetime, latest_available_target_datetime = normalize_datetimes(
        genesis_datetime=genesis_datetime,
        target_datetime=target_datetime,
        training_data=get_weather_data(
            bounding_box=bounding_box,
            start=round_datetime(dt=round_datetime_to_hour(dt=genesis_datetime), ceil=False),
            end=round_datetime(dt=round_datetime_to_hour(dt=target_datetime), ceil=True),
            # Make sure the processed weather data cache is invalidated every hour.
            fetch_date=datetime_now(),
        ),
    )
    latest_model, latest_scores, latest_training_instance = get_latest_model(
        bounding_box=bounding_box,
        target_datetime=latest_available_target_datetime,
        genesis_datetime=genesis_datetime,
        n_days_pretraining=n_days_pretraining,
        lookback_window=lookback_window,
    )
    forecast = get_forecast(
        latest_model=latest_model,
        latest_training_instance=latest_training_instance,
        target_datetime=target_datetime,
        forecast_window=forecast_window,
    )
    return forecast, latest_scores


# by default, set target and genesis datetime launchplan to three days ago.
DEFAULT_GENESIS_TIME = (pd.Timestamp.now().floor("d") - pd.Timedelta(days=3)).to_pydatetime()
DEFAULT_INPUTS = {
    "target_datetime": DEFAULT_GENESIS_TIME,
    "genesis_datetime": DEFAULT_GENESIS_TIME,
    "n_days_pretraining": 30,  # one month pre training
    "lookback_window": 24 * 3,  # 3-day lookback
    "forecast_window": 24 * 3,  # 3-day forecast
}

EMAIL_NOTIFICATION = Email(
    phases=[
        WorkflowExecutionPhase.SUCCEEDED,
        WorkflowExecutionPhase.TIMED_OUT,
        WorkflowExecutionPhase.FAILED,
    ],
    recipients_email=["niels@union.ai"]
)

SLACK_NOTIFICATION = Slack(
    phases=[
        WorkflowExecutionPhase.SUCCEEDED,
        WorkflowExecutionPhase.TIMED_OUT,
        WorkflowExecutionPhase.FAILED,
    ],
    recipients_email=["flytlab-notifications-aaaad6weta5ic55r7lmejgwzha@unionai.slack.com"],
)

# run the job every hour
CRON_SCHEDULE = CronSchedule(
    schedule="0 * * * *",
    kickoff_time_input_arg="target_datetime",
)

KWARGS = {
    "default_inputs": DEFAULT_INPUTS,
    "schedule": CRON_SCHEDULE,
    "notifications": [EMAIL_NOTIFICATION, SLACK_NOTIFICATION],
}

LOCATIONS = {
    "atlanta": "Atlanta, GA USA",
    "seattle": "Seattle, WA USA",
    "hyderabad": "Hyderabad, Telangana India",
    "mumbai": "Mumbai, MH India",
    "taipei": "Taipei, Taiwan",
    "appleton": "Green Bay, WI USA",
    "dharamshala": "Dharmsala, HP India",
    "fremont": "Fremont, CA USA",
}

for location_name, location_query in LOCATIONS.items():
    LaunchPlan.get_or_create(
        workflow=forecast_weather,
        name=f"{location_name}_weather_forecast_v2",
        fixed_inputs={"location_query": location_query},
        **KWARGS,
    )


if __name__ == "__main__":
    N_DAYS = 4
    N_HOURS = 24 * N_DAYS
    for location_query in LOCATIONS.values():
        print(f"location: {location_query}")
        genesis_datetime = (pd.Timestamp.now().floor("d") - pd.Timedelta(days=4)).to_pydatetime()
        target_datetime = (pd.Timestamp.now().floor("d") - pd.Timedelta(days=0)).to_pydatetime()
        forecast, scores = forecast_weather(
            location_query=location_query,
            target_datetime=target_datetime,
            genesis_datetime=genesis_datetime,
            n_days_pretraining=7,
            lookback_window=24 * 3,
            forecast_window=24 * 3,
        )
        print(f"Target Datetime: {target_datetime}")
        print("Forecasts:")
        for prediction in forecast.predictions[-5:]:
            print(prediction)
        print(f"Scores: {scores}\n--------")
