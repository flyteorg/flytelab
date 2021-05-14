import copy
import itertools
import joblib
import logging
import os
import time
from dataclasses import astuple
from datetime import datetime, timedelta
from typing import List

from sklearn.linear_model import SGDRegressor

from flytekit import task, dynamic, workflow, Resources
from flytekit.types.file import JoblibSerializedFile

from flytelab.weather_forecasting import data, trainer, types


logger = logging.getLogger(__file__)


MAX_RETRIES = 10


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


@task(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
def get_training_instance(location: str, target_date: datetime) -> data.TrainingInstance:
    logger.info(f"getting training/validation batches for target date {target_date}")
    for i in range(MAX_RETRIES):
        if i > 0:
            logger.info(f"retry {i}")
        try:
            instance = data.get_training_instance(location, target_date.date())
            return instance
        except RuntimeError:
            time.sleep(1)


@workflow
def get_training_instance_wf(location: str, target_date: datetime) -> data.TrainingInstance:
    return get_training_instance(location=location, target_date=target_date)


@task(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
def create_batch(
    training_data: List[data.TrainingInstance],
    validation_data: List[data.TrainingInstance]
) -> data.Batch:
    return data.Batch(training_data=training_data, validation_data=validation_data)


@dynamic(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
def get_training_data(now: datetime, location: str, config: types.Config) -> List[data.Batch]:
    batches = []
    genesis_to_now = (now.date() - config.model.genesis_date.date()).days + 1
    for i, n_days in enumerate(
        itertools.chain(
            # prior days knowledge lookback from the genesis date
            (-x for x in reversed(range(int(config.model.prior_days_window) + 1))),
            # update model from genesis date to current date
            range(1, genesis_to_now),
        )
    ):
        target_date = config.model.genesis_date + timedelta(days=n_days)
        logger.info(f"[batch {i}] getting training/validation batches for target date {target_date}")
        training_data: List[data.TrainingInstance] = [
            get_training_instance(location=location, target_date=target_date - timedelta(days=j))
            for j in range(int(config.model.batch_size))
        ]
        validation_data: List[data.TrainingInstance] = [
            get_training_instance(location=location, target_date=target_date + timedelta(days=j))
            for j in range(int(config.model.validation_size))
        ]
        batches.append(create_batch(training_data=training_data, validation_data=validation_data))
    return batches


@task(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
def update_model(model_file: JoblibSerializedFile, batch: data.Batch) -> JoblibSerializedFile:
    if trainer.stop_training(batch.training_data):
        return model_file

    with open(model_file, "rb") as f:
        model = joblib.load(f)

    model = trainer.update_model(model, batch.training_data)
    model_id = joblib.hash(model)
    out = f"/tmp/{model_id}.joblib"
    with open(out, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=out)


@dynamic(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
def get_latest_model(now: datetime, batches: List[data.Batch]) -> JoblibSerializedFile:
    # TODO: need to figure out how to make these parameterized so that
    # training picks up from yesterday
    model = SGDRegressor(
        penalty="l1",
        alpha=1.0,
        random_state=567,
        warm_start=True,
        early_stopping=False,
    )
    model_id = joblib.hash(model)
    out = f"/tmp/{model_id}.joblib"
    with open(out, "wb") as f:
        joblib.dump(model, f, compress=True)
    model_file = JoblibSerializedFile(path=out)
    for batch in batches:
        model_file = update_model(model_file=model_file, batch=batch)
    return model_file


@task(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
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
            features = predictions[:len(features)]

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


@task(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
def create_forecast(target_date: datetime, model_id: str, predictions: List[types.Prediction]) -> types.Forecast:
    return types.Forecast(created_at=target_date, model_id=model_id, predictions=predictions)


@dynamic(requests=Resources(cpu="2", mem="500Mi"), limits=Resources(cpu="2", mem="1000Mi"))
def get_forecast(
    location: str,
    target_date: datetime,
    model_file: JoblibSerializedFile,
    config: types.Config
) -> types.Forecast:
    with open(model_file, "rb") as f:
        model = joblib.load(f)

    predictions: List[types.Prediction] = []
    for i, n_days in enumerate(range(int(config.forecast.n_days) + 1)):
        forecast_date = target_date + timedelta(days=n_days)
        forecast_batch = [
            get_training_instance(location=location, target_date=forecast_date - timedelta(days=j))
            for j in range(1, int(config.model.batch_size) + 1)
        ]
        pred = get_prediction(model_file=model_file, forecast_batch=forecast_batch, predictions=predictions)
        logger.info(f"[forecasting {i}] {pred}")
        predictions.insert(0, pred)  # most recent forecasts first

    return create_forecast(target_date=target_date, model_id=str(joblib.hash(model)), predictions=predictions)


@workflow
def run_pipeline(
    location: str,
    model_genesis_date: datetime = datetime.now(),
    model_prior_days_window: int = 3,
    model_batch_size: int = 7,
    model_validation_size: int = 1,
    instance_lookback_window: int = 3,
    instance_n_year_lookback: int = 1,
    metrics_scorers: List[str] = ["neg_mean_absolute_error", "neg_mean_squared_error"],
    forecast_n_days: int = 3,
) -> types.Forecast:
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
    now = datetime.now()
    batches = get_training_data(
        now=now,
        location=location,
        config=config
    )
    model_file = get_latest_model(now=now, batches=batches)
    return get_forecast(
        location=location,
        target_date=now,
        model_file=model_file,
        config=config,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s:: %(message)s")
    forecast = run_pipeline(location="Atlanta, GA US")
    logger.info("forecast")
    logger.info(forecast)
