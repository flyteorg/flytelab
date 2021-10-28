import base64
import math
import logging
import os
import time
from io import BytesIO, StringIO
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

from flytekit import conditional, dynamic, task, workflow, kwtypes, CronSchedule, LaunchPlan, Resources, Slack, Email
from flytekit.models.core.execution import WorkflowExecutionPhase
from flytekit.types.file import JoblibSerializedFile
from flytekit.types.schema import FlyteSchema

import flytekitplugins.pandera


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
# logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")

geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=10)


TrainingSchema = FlyteSchema[kwtypes(date=datetime, air_temp=float, dew_temp=float)]


class GlobalHourlyDataRaw(pa.SchemaModel):
    date: Series[pa.typing.DateTime]
    tmp: Series[str]

    class Config:
        coerce = True


class GlobalHourlyDataClean(pa.SchemaModel):
    date: Index[pa.typing.DateTime]
    air_temp: Series[float]
    dew_temp: Series[float]

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


@dataclass_json
@dataclass
class ModelUpdate:
    model_file: str
    scores: Scores
    training_instance: TrainingInstance


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
def parse_global_hourly_data(df: DataFrame[GlobalHourlyDataRaw]) -> DataFrame[GlobalHourlyDataClean]:
    """Process raw global hourly data.

    For reference, see data documents:
    - https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
    - https://www.ncei.noaa.gov/data/global-hourly/doc/CSV_HELP.pdf
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["air_temp", "air_temp_quality_code", "date"])
    return (
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
        .loc[lambda df: ~df.air_temp_quality_code.isin({"3", "7"})]
        # get hourly mean air temperature across readings of different sensor locations
        .groupby("date").agg({"air_temp": "mean", "dew_temp": "mean"})
        # resample at hourly interval to make sure all hourly intervals have an entry
        .resample("1H").mean()
        # linearly interpolate into the future
        .interpolate(method="linear", limit_direction="forward", limit=None)
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
def get_weather_data(
    bounding_box: BoundingBox,
    start: datetime,
    end: datetime,
    fetch_date: datetime,
) -> TrainingSchema:
    logger.info(
        f"getting global hourly data for query: {bounding_box} between {start} and {end}, fetch date: {fetch_date}"
    )
    responses = get_global_hourly_data_responses(bounding_box=bounding_box, start=start, end=end)
    data = process_raw_training_data(data=get_raw_data(responses=responses))
    training_data = TrainingSchema()
    training_data.open().write(data)
    return training_data


def _prepare_training_instance(training_data: TrainingSchema, start: datetime, end: datetime) -> TrainingInstance:
    logging.info(f"get training instance: {start} - {end}")

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
        air_temp_features=features["air_temp"].tolist()
        dew_temp_features=features["dew_temp"].tolist()
        time_based_feature=features.index[0].to_pydatetime()
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
def prepare_training_instance(training_data: TrainingSchema, start: datetime, end: datetime) -> TrainingInstance:
    return _prepare_training_instance(training_data.open().all(), start, end)


@task
def round_datetime(dt: datetime, ceil: bool) -> datetime:
    """Round date-times to the nearest hour."""
    return datetime(dt.year + 1 if ceil else dt.year, 1, 1, 0, tzinfo=None)


@task(requests=request_resources, limits=limit_resources)
def instances_from_pretraining_data(
    training_data: TrainingSchema,
    pretraining_start_date: datetime,
    genesis_datetime: datetime,
    lookback_window: int,
) -> List[TrainingInstance]:
    training_data: pd.Dataframe = training_data.open().all()
    training_instances = []
    diff_in_hours = (genesis_datetime - pretraining_start_date).days * 24
    for i in range(1, diff_in_hours + 1):
        current_datetime = pretraining_start_date + timedelta(hours=i)
        training_instance = _prepare_training_instance(
            training_data=training_data,
            start=current_datetime - timedelta(hours=lookback_window),
            end=current_datetime,
        )
        training_instances.append(training_instance)
    return training_instances


@dynamic(
    requests=request_resources,
    limits=limit_resources,
)
def get_pretraining_instances(
    bounding_box: BoundingBox,
    genesis_datetime: datetime,
    n_days_pretraining: int,
    lookback_window: int,
) -> List[TrainingInstance]:

    genesis_datetime = genesis_datetime.replace(tzinfo=None)
    pretraining_start_date = genesis_datetime - timedelta(days=n_days_pretraining)
    training_data = get_weather_data(
        bounding_box=bounding_box,
        start=round_datetime(dt=pretraining_start_date, ceil=False),
        end=round_datetime(dt=genesis_datetime, ceil=True),
        # Make sure the processed weather data cache is invalidated every hour.
        fetch_date=pd.Timestamp.now().floor("H").to_pydatetime(),
    )
    return instances_from_pretraining_data(
        training_data=training_data,
        pretraining_start_date=pretraining_start_date,
        genesis_datetime=genesis_datetime,
        lookback_window=lookback_window,
    )


def serialize_model(model: BaseEstimator) -> str:
    """Convert model object to compressed byte string."""
    buffer = BytesIO()
    joblib.dump(model, buffer, compress=True)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def deserialize_model(serialized_model: str) -> BaseEstimator:
    """Load model object from compressed byte string."""
    return joblib.load(BytesIO(base64.b64decode(serialized_model.encode())))


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
        scores.valid_exp_mae
    )
    logging.info(f"updated model: train mae={train_exp_mae}, valid mae={valid_exp_mae}")
    return model, Scores(train_exp_mae, valid_exp_mae)


@task
def pretrain_model(model: str, scores: Scores, training_instances: List[TrainingInstance]) -> ModelUpdate:
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
    return pretrain_model(
        model=serialize_model(model),
        scores=Scores(),
        training_instances=get_pretraining_instances(
            bounding_box=bounding_box,
            genesis_datetime=genesis_datetime,
            n_days_pretraining=n_days_pretraining,
            lookback_window=lookback_window,
        ),
    )


@dynamic(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def get_training_instance(bounding_box: BoundingBox, start: datetime, end: datetime) -> TrainingInstance:
    """Gets a single training instance.
    
    Before getting the raw weather data, round the start and end date so that the result can be cached.
    """
    logging.info(f"get training instance: {start} - {end}")
    return prepare_training_instance(
        training_data=get_weather_data(
            bounding_box=bounding_box,
            start=round_datetime(dt=start, ceil=False),
            end=round_datetime(dt=end, ceil=True),
            # Make sure the processed weather data cache is invalidated every hour.
            fetch_date=pd.Timestamp.now().floor("H").to_pydatetime(),
        ),
        start=start,
        end=end,
    )


@task(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def update_model(
    prev_model_update: ModelUpdate,
    training_instance: TrainingInstance,
) -> ModelUpdate:
    model, scores = _update_model(
        deserialize_model(prev_model_update.model_file), prev_model_update.scores, training_instance
    )
    return ModelUpdate(model_file=serialize_model(model), scores=scores, training_instance=training_instance)


@task
def previous_timestep(dt: datetime) -> datetime:
    return dt - timedelta(seconds=60 * 60)


@dynamic(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def get_updated_model_recursively(
    bounding_box: BoundingBox,
    target_datetime: datetime,
    genesis_datetime: datetime,
    n_days_pretraining: int,
    lookback_window: int,
)  -> ModelUpdate:
    prev_model_update = get_latest_model(
        bounding_box=bounding_box,
        target_datetime=previous_timestep(dt=target_datetime),
        genesis_datetime=genesis_datetime,
        n_days_pretraining=n_days_pretraining,
        lookback_window=lookback_window,
    )
    # TODO: if target_datetime is more than one timestep greater than prev_model_update,
    # then do multi-instance update
    training_instance = get_training_instance(
        bounding_box=bounding_box,
        start=target_datetime - timedelta(hours=lookback_window),
        end=target_datetime,
    )
    return update_model(
        prev_model_update=prev_model_update,
        training_instance=training_instance,
    )

@workflow
def get_latest_model(
    bounding_box: BoundingBox,
    target_datetime: datetime,
    genesis_datetime: datetime,
    n_days_pretraining: int,
    lookback_window: int,
) -> ModelUpdate:
    return (
        conditional("init_model")
        .if_(genesis_datetime == target_datetime)
        .then(
            init_model(
                bounding_box=bounding_box,
                genesis_datetime=genesis_datetime,
                n_days_pretraining=n_days_pretraining,
                lookback_window=lookback_window,
            )
        )
        .else_()
        .then(
            get_updated_model_recursively(
                bounding_box=bounding_box,
                target_datetime=target_datetime,
                genesis_datetime=genesis_datetime,
                n_days_pretraining=n_days_pretraining,
                lookback_window=lookback_window,
            )
        )
    )


@task(
    requests=request_resources,
    limits=limit_resources,
)
def get_forecast(
    latest_model_update: ModelUpdate,
    target_datetime: datetime,
    forecast_window: int,
) -> Forecast:
    model = deserialize_model(latest_model_update.model_file)
    latest_training_instance = latest_model_update.training_instance

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


def round_datetime_to_hour(dt: datetime) -> datetime:
    """Round date-times to the nearest hour."""
    return datetime(dt.year, dt.month, dt.day, hour=dt.hour, tzinfo=None)


@task(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def normalize_datetimes(
    bounding_box: BoundingBox,
    genesis_datetime: datetime,
    target_datetime: datetime,
) -> NamedTuple("NormalizedDatetimes", genesis_datetime=datetime, target_datetime=datetime):
    """Get the date of the latest available training data."""
    genesis_datetime = round_datetime_to_hour(genesis_datetime)
    target_datetime = round_datetime_to_hour(target_datetime)
    [response, *_], _ = call_noaa_api(
        bounding_box=bounding_box,
        start=datetime(genesis_datetime.year, 1, 1, 0, tzinfo=None),
        end=target_datetime,
        responses=[]
    )
    latest_available = pd.to_datetime(response["endDate"]).floor("H").to_pydatetime()
    target_datetime = min(target_datetime, latest_available)
    if target_datetime < genesis_datetime:
        genesis_datetime = latest_available
    return genesis_datetime, target_datetime


@task
def get_scores(latest_model_update: ModelUpdate) -> Scores:
    return latest_model_update.scores


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
    genesis_datetime, target_datetime = normalize_datetimes(
        bounding_box=bounding_box,
        genesis_datetime=genesis_datetime,
        target_datetime=target_datetime,
    )
    latest_model_update = get_latest_model(
        bounding_box=get_bounding_box(location_query=location_query),
        target_datetime=target_datetime,
        genesis_datetime=genesis_datetime,
        n_days_pretraining=n_days_pretraining,
        lookback_window=lookback_window,
    )
    forecast = get_forecast(
        latest_model_update=latest_model_update,
        target_datetime=target_datetime,
        forecast_window=forecast_window,
    )
    return forecast, get_scores(latest_model_update=latest_model_update)


# by default, set target and genesis datetime launchplan to previous day.
NOW = (pd.Timestamp.now().floor("H") - pd.Timedelta(days=1)).to_pydatetime()
DEFAULT_INPUTS = {
    "target_datetime": NOW,
    "genesis_datetime": NOW,
    "n_days_pretraining": 14,
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
}

for location_name, location_query in LOCATIONS.items():
    LaunchPlan.get_or_create(
        workflow=forecast_weather,
        name=f"{location_name}_weather_forecast_v2",
        fixed_inputs={"location_query": location_query},
        **KWARGS,
    )


if __name__ == "__main__":
    for location_query in LOCATIONS.values():
        print(f"location: {location_query}")
        forecast, scores = forecast_weather(
            location_query=location_query,
            target_datetime=datetime.now() - timedelta(days=0),
            genesis_datetime=datetime.now() - timedelta(days=2),
            n_days_pretraining=7,
            lookback_window=24 * 3,
            forecast_window=24 * 3,
        )
        print("Forecasts:")
        for prediction in forecast.predictions[:3]:
            print(prediction)
        print(f"Scores: {scores}\n--------")
