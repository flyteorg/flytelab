import copy
import itertools
import joblib
import json
import logging
import os
import time
from dataclasses import astuple
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from sklearn.linear_model import SGDRegressor

from flytekit import task, dynamic, workflow
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


@task(cache=True, cache_version="1")
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


@dynamic(cache=True, cache_version="1")
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
        training_data = [
            get_training_instance(location=location, target_date=target_date - timedelta(days=j))
            for j in range(int(config.model.batch_size))
        ]
        validation_data = [
            instance for instance in (
                get_training_instance(location=location, target_date=target_date + timedelta(days=j + 1))
                for j in range(int(config.model.validation_size))
            )
            if not pd.isna(instance.target)
        ]
        batches.append(data.Batch(training_data=training_data, validation_data=validation_data))
    return batches


@task(cache=True, cache_version="1")
def update_model(
    model_file: JoblibSerializedFile,
    batch: data.Batch,  # type: ignore
) -> JoblibSerializedFile:  # type: ignore
    if model_file.path != "" and trainer.stop_training(batch.training_data):
        return model_file

    if model_file.path == "":
        model = SGDRegressor(
            penalty="l1",
            alpha=1.0,
            random_state=567,
            warm_start=True,
            early_stopping=False,
        )
    else:
        with open(model_file, "rb") as f:
            model = joblib.load(f)

    model = trainer.update_model(model, batch.training_data)
    model_id = joblib.hash(model)
    out = f"/tmp/{model_id}.joblib"
    with open(out, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=out)


@dynamic(cache=True, cache_version="1")
def get_latest_model(batches: List[data.Batch]) -> JoblibSerializedFile:  # type: ignore
    # TODO: need to figure out how to make these parameterized so that
    # training picks up from yesterday
    model_file = JoblibSerializedFile(path="")
    for batch in batches:
        model_file = update_model(model_file=model_file, batch=batch)
    return model_file


@task(cache=True, cache_version="1")
def get_prediction(
    model_file: JoblibSerializedFile, forecast_batch: List[data.TrainingInstance], predictions: List[float]
) -> float:
    """Replaces the null value future placeholders with the forecasts so far."""
    with open(model_file, "rb") as f:
        model = joblib.load(f)

    def _fill_nans(instance, predictions):
        features = copy.copy(instance.features)
        if len(predictions) <= len(features):
            for i, f in enumerate(predictions):
                features[i] = f
        else:
            features = predictions[:len(features)]
        return features

    if predictions:
        forecast_batch = [
            data.TrainingInstance(_fill_nans(instance, predictions[i:]), *astuple(instance)[1:])
            for i, instance in enumerate(forecast_batch)
        ]

    try:
        features, _ = trainer.batch_to_norm_vectors(forecast_batch)
        pred = model.predict(features[:1, :]).item()
    except Exception:
        import ipdb; ipdb.set_trace()
        pred = None
    return pred


@dynamic(cache=True, cache_version="1")
def get_forecast(
    location: str,
    target_date: datetime,
    model_file: JoblibSerializedFile,
    config: types.Config
) -> types.Forecast:
    with open(model_file, "rb") as f:
        model = joblib.load(f)

    predictions, forecast_dates = [], []
    for i, n_days in enumerate(range(int(config.forecast.n_days) + 1)):
        forecast_date = target_date + timedelta(days=n_days)
        forecast_batch = [
            get_training_instance(location=location, target_date=forecast_date - timedelta(days=j))
            for j in range(1, int(config.model.batch_size) + 1)
        ]
        pred = get_prediction(model_file=model_file, forecast_batch=forecast_batch, predictions=predictions)
        logger.info(f"[forecasting {i}] for target_date: {forecast_date}, prediction: {pred}")
        # most recent forecasts first
        predictions.insert(0, pred)
        forecast_dates.insert(0, forecast_date)
        logger.info("=" * 50)

    return types.Forecast(
        created_at=target_date,
        model_id=joblib.hash(model),
        predictions=[types.Prediction(value=p, date=d) for p, d in zip(predictions, forecast_dates)]
    )


@workflow
def run_pipeline(config: types.Config, location: str) -> types.Forecast:
    logger.info(f"training model with config:\n{json.dumps(config, indent=4, default=str)}")
    now = datetime.now()
    batches = get_training_data(
        now=now,
        location=location,
        config=config
    )
    model_file = get_latest_model(batches=batches)
    return get_forecast(
        location=location,
        target_date=now,
        model_file=model_file,
        config=config,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s:: %(message)s")
    from datetime import date

    config = types.Config(
        model=types.ModelConfig(
            genesis_date=date(2021, 5, 11),
            # lookback window for pre-training the model n number of days before the genesis date.
            prior_days_window=1,
            batch_size=3,
            validation_size=1,
        ),
        instance=types.InstanceConfig(
            lookback_window=3,
            n_year_lookback=1,
        ),
        metrics=types.MetricsConfig(
            scorers=["neg_mean_absolute_error", "neg_mean_squared_error"],
        ),
        forecast=types.ForecastConfig(
            n_days=3,
        )
    )
    forecast = run_pipeline(config=config, location="Atlanta, GA US")
    logger.info("forecast")
    logger.info(forecast)
