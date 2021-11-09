"""
NOTE: This version of the weather forecasting workflow is no longer maintained
"""

import copy
import itertools
import joblib
import logging
import os
import pandas as pd
import time
from dataclasses import astuple
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, NamedTuple

from sklearn.linear_model import SGDRegressor

from flytekit import task, dynamic, workflow, CronSchedule, LaunchPlan, Resources, Slack
from flytekit.models.core.execution import WorkflowExecutionPhase
from flytekit.types.file import JoblibSerializedFile

from flytelab.weather_forecasting.v1 import cache, data, trainer, types


logger = logging.getLogger(__file__)


MAX_RETRIES = 10
CACHE_VERSION = "2"

request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")

TrainingData = NamedTuple("TrainingData", batches=List[data.Batch], imputation_date=datetime)


def floor_date(dt: datetime):
    return datetime(dt.year, dt.month, dt.day, tzinfo=None)

def to_datetime(date: date):
    return datetime(date.year, date.month, date.day)


@task
def fetch_key(key: str) -> str:
    print("fetching API key")
    return os.getenv(key)


@workflow
def get_api_key(key: str) -> str:
    return fetch_key(key=key)


@task
def get_config(
    model_genesis_date: datetime,
    model_prior_days_window: int,
    model_batch_size: int,
    model_validation_size: int,
    instance_lookback_window: int,
    instance_n_year_lookback: int,
    metrics_scorers: List[str],
    forecast_n_days: int,
) -> types.Config:
    return types.Config(
        model=types.ModelConfig(
            genesis_date=model_genesis_date,
            # lookback window for pre-training the model n number of days before the genesis date.
            prior_days_window=model_prior_days_window,
            batch_size=model_batch_size,
            validation_size=model_validation_size,
        ),
        instance=types.InstanceConfig(
            lookback_window=instance_lookback_window,
            n_year_lookback=instance_n_year_lookback,
        ),
        metrics=types.MetricsConfig(
            scorers=metrics_scorers,
        ),
        forecast=types.ForecastConfig(
            n_days=forecast_n_days,
        )
    )


def get_instance(location: str, target_date: datetime, instance_config: types.InstanceConfig) -> data.TrainingInstance:
    logger.info(f"getting training/validation batches for target date {target_date}")
    for i in range(MAX_RETRIES):
        try:
            return data.get_training_instance(
                location,
                target_date.date(),
                lookback_window=int(instance_config.lookback_window),
                n_year_lookback=int(instance_config.n_year_lookback),
                instance_id=cache.create_id(target_date, location.encode(), str(instance_config).encode())
            )
        except Exception as exc:
            logger.info(f"error on retry {i}: {exc}")
            time.sleep(1)


@task(
    retries=30,
    timeout=600,
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources
)
def get_training_instance(
    location: str,
    target_date: datetime,
    instance_config: types.InstanceConfig
) -> data.TrainingInstance:
    return get_instance(location, target_date, instance_config)


@task(retries=30, timeout=600, requests=request_resources, limits=limit_resources)
def get_partial_instance(
    location: str,
    target_date: datetime,
    instance_config: types.InstanceConfig
) -> data.TrainingInstance:
    return get_instance(location, target_date, instance_config)


@workflow
def get_training_instance_wf(
    location: str,
    target_date: datetime,
    instance_config: types.InstanceConfig = types.InstanceConfig(
        lookback_window=7,
        n_year_lookback=1,
    )
) -> data.TrainingInstance:
    return get_training_instance(location=location, target_date=target_date, instance_config=instance_config)


@task(requests=request_resources, limits=limit_resources)
def create_batch(
    training_data: List[data.TrainingInstance],
    validation_data: List[data.TrainingInstance]
) -> data.Batch:
    return data.Batch(
        training_data=[x for x in training_data if x is not None and not pd.isna(x.target)],
        validation_data=[x for x in validation_data if not x is not None and pd.isna(x.target)],
    )


@task(requests=request_resources, limits=limit_resources)
def prepare_training_data(batches: List[data.Batch]) -> TrainingData:
    """Stop training when the most recent instance has a null target."""
    output_batches: List[data.Batch] = []
    last_instance = batches[-1].training_data[0]
    imputation_date = last_instance.target_date

    for batch in batches:
        if pd.isna(batch.training_data[0]):
            # exclude batches from where the target of the first record in the training data is null
            imputation_date = batch.training_data[0].target_date
            break
        output_batches.append(batch)

    return output_batches, to_datetime(imputation_date)


@dynamic(requests=request_resources, limits=limit_resources)
def get_training_data(now: datetime, location: str, config: types.Config) -> TrainingData:
    batches = []
    now = floor_date(now)
    genesis_datetime = floor_date(config.model.genesis_date)
    genesis_to_now = (now.date() - genesis_datetime.date()).days + 1
    # imputation date indicates when we need to start imputing data since the primary data source
    # doesn't have temperature data starting from this date.
    for i, n_days in enumerate(
        itertools.chain(
            # prior days knowledge lookback from the genesis date
            (-x for x in reversed(range(int(config.model.prior_days_window) + 1))),
            # update model from genesis date to current date
            range(1, genesis_to_now),
        )
    ):
        target_date = genesis_datetime + timedelta(days=n_days)
        logger.info(f"[batch {i}] getting training/validation batches for target date {target_date}")
        training_data: List[data.TrainingInstance] = [
            get_training_instance(
                location=location,
                target_date=target_date - timedelta(days=j),
                instance_config=config.instance
            )
            if (target_date - timedelta(days=j)) < now
            else get_partial_instance(
                location=location,
                target_date=target_date - timedelta(days=j),
                instance_config=config.instance
            )
            for j in range(int(config.model.batch_size))
        ]
        validation_data: List[data.TrainingInstance] = [
            get_partial_instance(
                location=location,
                target_date=target_date + timedelta(days=j),
                instance_config=config.instance
            )
            for j in range(int(config.model.validation_size))
        ]
        batches.append(create_batch(training_data=training_data, validation_data=validation_data))
    return prepare_training_data(batches=batches)


@task(cache=True, cache_version=CACHE_VERSION, requests=request_resources, limits=limit_resources)
def init_model(
    genesis_date: datetime,
    model_config: types.ModelConfig,
    instance_config: types.InstanceConfig
) -> JoblibSerializedFile:
    # TODO: model hyperparameters should be in the config
    model = SGDRegressor(
        penalty="l1",
        alpha=1.0,
        random_state=567,
        warm_start=True,
        early_stopping=False,
    )
    model_id = cache.create_model_id(
        genesis_date=genesis_date.date(),
        update_date=genesis_date.date(),
        model_config=model_config,
        instance_config=instance_config,
    )
    out_file = f"/tmp/{model_id}.joblib"
    with open(out_file, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=out_file)


@task(cache=True, cache_version=CACHE_VERSION, requests=request_resources, limits=limit_resources)
def update_model(
    model_file: JoblibSerializedFile,
    batch: data.Batch,
    model_config: types.ModelConfig,
    instance_config: types.InstanceConfig,
    metrics_config: types.MetricsConfig,
) -> (JoblibSerializedFile, List[types.Metrics]):  # type: ignore
    with open(model_file, "rb") as f:
        model = joblib.load(f)

    model = trainer.update_model(model, batch.training_data)
    metrics = trainer.evaluate_model(metrics_config.scorers, model, batch.training_data, batch.validation_data)
    model_id = cache.create_model_id(
        genesis_date=model_config.genesis_date.date(),
        # the first element in the training data should be the most recent training example
        update_date=batch.training_data[0].target_date,
        model_config=model_config,
        instance_config=instance_config,        
        instances=batch.training_data,
    )
    out = f"/tmp/{model_id}.joblib"
    with open(out, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=out), metrics


@dynamic(cache=True, cache_version=CACHE_VERSION, requests=request_resources, limits=limit_resources)
def get_latest_model(
    config: types.Config,
    batches: List[data.Batch]
) -> (JoblibSerializedFile, List[List[types.Metrics]]):  # type: ignore
    model_file = init_model(
        genesis_date=config.model.genesis_date,
        model_config=config.model,
        instance_config=config.instance,
    )
    metrics_per_batch = []
    for batch in batches:
        model_file, metrics = update_model(
            model_file=model_file,
            batch=batch,
            model_config=config.model,
            instance_config=config.instance,
            metrics_config=config.metrics,
        )
        metrics_per_batch.append(metrics)
    return model_file, metrics_per_batch


@task(requests=request_resources, limits=limit_resources)
def get_prediction(
    model_file: JoblibSerializedFile,
    forecast_batch: List[data.TrainingInstance],
    predictions: List[types.Prediction],
) -> types.Prediction:
    """Replaces the null value future placeholders with the forecasts so far."""
    with open(model_file, "rb") as f:
        model = joblib.load(f)

    def new_instance(instance: data.TrainingInstance, predictions: List[types.Prediction]):
        features = copy.copy(instance.features)
        if len(predictions) <= len(features):
            for i, pred in enumerate(predictions):
                features[i] = pred.value
        else:
            features = [p.value for p in predictions[:len(features)]]

        return data.TrainingInstance(features, *astuple(instance)[1:])

    if predictions:
        forecast_batch = [new_instance(instance, predictions[i:]) for i, instance in enumerate(forecast_batch)]

    error = None
    try:
        features, _ = trainer.batch_to_norm_vectors(forecast_batch)
        # only generate prediction for last item
        pred = model.predict(features[:1, :]).item()
    except Exception as exc:
        pred = None
        error = str(exc)
    return types.Prediction(value=pred, error=error, date=forecast_batch[0].target_date)


@task(requests=request_resources, limits=limit_resources)
def create_forecast(target_date: datetime, model_id: str, predictions: List[types.Prediction]) -> types.Forecast:
    return types.Forecast(created_at=target_date, model_id=model_id, predictions=predictions)


@dynamic(requests=request_resources, limits=limit_resources)
def get_forecast(
    location: str,
    now: datetime,
    imputation_date: datetime,
    model_file: JoblibSerializedFile,
    config: types.Config
) -> types.Forecast:
    # set time components to 0
    now = floor_date(now)
    predictions: List[types.Prediction] = []

    # NOTE: this workflow assumes datetimes are timezone-unaware
    n_imputations = (now - floor_date(imputation_date)).days + 1
    for i, n_days in enumerate(range(int(config.forecast.n_days) + n_imputations + 1)):
        forecast_date = floor_date(imputation_date + timedelta(days=n_days))
        forecast_batch = [
            get_training_instance(
                location=location,
                target_date=forecast_date - timedelta(days=j),
                instance_config=config.instance
            )
            if (forecast_date - timedelta(days=j)) < now
            else get_partial_instance(
                location=location,
                target_date=forecast_date - timedelta(days=j),
                instance_config=config.instance
            )
            for j in range(1, int(config.model.batch_size) + 1)
        ]
        pred = get_prediction(model_file=model_file, forecast_batch=forecast_batch, predictions=predictions)
        logger.info(f"[forecasting {i}] {pred}")
        predictions.insert(0, pred)  # most recent forecasts first

    return create_forecast(
        target_date=now,
        model_id=Path(model_file).stem,
        # don't return imputations
        predictions=predictions[:-n_imputations],
    )


@workflow
def forecast_weather(
    location: str,
    model_genesis_date: datetime = datetime.now(),
    model_prior_days_window: int = 1,
    model_batch_size: int = 7,
    model_validation_size: int = 1,
    instance_lookback_window: int = 3,
    instance_n_year_lookback: int = 1,
    metrics_scorers: List[str] = ["neg_mean_absolute_error", "neg_mean_squared_error"],
    forecast_n_days: int = 3,
) -> NamedTuple("WeatherForecast", [("forecast", types.Forecast), ("metrics", List[List[types.Metrics]])]):
    config = get_config(
        model_genesis_date=model_genesis_date,
        model_prior_days_window=model_prior_days_window,
        model_batch_size=model_batch_size,
        model_validation_size=model_validation_size,
        instance_lookback_window=instance_lookback_window,
        instance_n_year_lookback=instance_n_year_lookback,
        metrics_scorers=metrics_scorers,
        forecast_n_days=forecast_n_days,
    )
    logger.info("training model with config:")
    logger.info(config)
    now = floor_date(datetime.now())
    batches, imputation_date = get_training_data(now=now, location=location, config=config)
    model_file, metrics_per_batch = get_latest_model(config=config, batches=batches)
    forecast = get_forecast(
        location=location,
        now=now,
        imputation_date=imputation_date,
        model_file=model_file,
        config=config,
    )
    return forecast, metrics_per_batch


################
# Launch Plans #
################

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
DEFAULT_INPUTS = {
    "model_genesis_date": datetime(2021, 6, 1),
    "model_prior_days_window": 30,
    "instance_lookback_window": 30,
    "instance_n_year_lookback": 3,
    "forecast_n_days": 7,
}

atlanta_lp = LaunchPlan.get_or_create(
    workflow=forecast_weather,
    name="atlanta_weather_forecast",
    default_inputs=DEFAULT_INPUTS,
    fixed_inputs={"location": "Atlanta, GA USA"},
    schedule=CronSchedule("0 4 * * ? *"),  # EST midnight
    notifications=[SLACK_NOTIFICATION],
)

seattle_lp = LaunchPlan.get_or_create(
    workflow=forecast_weather,
    name="seattle_weather_forecast",
    default_inputs=DEFAULT_INPUTS,
    fixed_inputs={"location": "Seattle, WA USA"},
    schedule=CronSchedule("0 7 * * ? *"),  # PST midnight
    notifications=[SLACK_NOTIFICATION],
)

hyderabad_lp = LaunchPlan.get_or_create(
    workflow=forecast_weather,
    name="hyderabad_weather_forecast",
    default_inputs=DEFAULT_INPUTS,
    fixed_inputs={"location": "Hyderabad, Telangana India"},
    schedule=CronSchedule("30 18 * * ? *"),  # IST midnight
    notifications=[SLACK_NOTIFICATION],
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s:: %(message)s")
    forecast = forecast_weather(
        location="Atlanta, GA USA",
        model_genesis_date=datetime(2021, 6, 22),
        model_prior_days_window=3,
        instance_lookback_window=7,
        instance_n_year_lookback=1,
        forecast_n_days=7,
    )
    logger.info("forecast")
    logger.info(forecast)
