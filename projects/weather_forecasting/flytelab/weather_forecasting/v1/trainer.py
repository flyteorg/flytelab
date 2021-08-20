import copy
import datetime
import json
import itertools
import logging
from dataclasses import astuple, asdict
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import SCORERS

from flytelab.weather_forecasting.v1 import data, cache, types


def load_data(get_training_instance, location_query, target_date, config):
    training_batch = [
        get_training_instance(location_query, target_date - datetime.timedelta(days=i))
        for i in range(config.model.batch_size)
    ]
    validation_batch = [
        instance for instance in (
            get_training_instance(location_query, target_date + datetime.timedelta(days=i + 1))
            for i in range(config.model.validation_size)
        )
        if not pd.isna(instance.target)
    ]
    return training_batch, validation_batch


def stop_training(training_batch: List[data.TrainingInstance]) -> bool:
    for instance in training_batch:
        if pd.isna(instance.target):
            return True
    return False


def batch_to_vectors(batch: List[data.TrainingInstance]):
    features, target = [], []
    for instance in batch:
        features.append(instance.features)
        target.append(instance.target)

    # TODO:
    # - add timestamp features
    # - add additional weather data features
    # - implement streaming mean/std statistics for normalization

    target = np.array(target, dtype=np.float32)
    return features, target


def batch_to_norm_vectors(
    training_batch: List[data.TrainingInstance],
    validation_batch: Optional[List[data.TrainingInstance]] = None
):
    train_features, train_target = batch_to_vectors(training_batch)

    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)

    # normalize features
    train_features = (train_features - mean) / std

    if validation_batch is not None:
        validation_batch = [instance for instance in validation_batch if not pd.isna(instance.target)]
        if len(validation_batch) == 0:
            validation_features, validation_target = None, None
        else:
            validation_features, validation_target = batch_to_vectors(validation_batch)
            validation_features = (validation_features - mean) / std
        return train_features, train_target, validation_features, validation_target

    return train_features, train_target


def update_model(
    model: BaseEstimator,
    training_batch: List[data.TrainingInstance],
) -> BaseEstimator:
    return model.fit(*batch_to_norm_vectors(training_batch))


def evaluate_model(
    scorers: List[str],
    model: BaseEstimator,
    training_batch: List[data.TrainingInstance],
    validation_batch: List[data.TrainingInstance],
):
    train_features, train_target, validation_features, validation_target = batch_to_norm_vectors(
        training_batch, validation_batch
    )
    metrics = []
    for scorer in scorers:
        scoring_fn = SCORERS[scorer]
        metrics.append(
            types.Metrics(
                name=scorer,
                train=scoring_fn(model, train_features, train_target),
                train_size=len(training_batch),
                validation=(
                    scoring_fn(model, validation_features, validation_target)
                    if validation_features is not None and validation_target is not None
                    else None
                ),
                validation_size=len(validation_batch),
            )
        )
    return metrics


def prepare_forecast_batch(
    training_batch: List[data.TrainingInstance], predictions: List[float]
) -> List[data.TrainingInstance]:
    """Replaces the null value future placeholders with the forecasts so far."""
    if not predictions:
        return training_batch

    def _fill_nans(instance, predictions):
        features = copy.copy(instance.features)
        if len(predictions) <= len(features):
            for i, f in enumerate(predictions):
                features[i] = f
        else:
            features = predictions[:len(features)]
        return features

    forecast_batch = []
    for i, instance in enumerate(training_batch):
        forecast_batch.append(
            data.TrainingInstance(_fill_nans(instance, predictions[i:]), *astuple(instance)[1:])
        )

    return forecast_batch


def forecast_today(model: BaseEstimator, training_batch: List[data.TrainingInstance]):
    features, _ = batch_to_norm_vectors(training_batch)
    prediction = model.predict(features[:1, :]).item()  # just predict target_date
    return prediction


if __name__ == "__main__":

    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(asctime)s:: %(message)s")
    logger = logging.getLogger(__file__)

    config = types.Config(
        model=types.ModelConfig(
            genesis_date=datetime.date(2021, 4, 20),
            # lookback window for pre-training the model n number of days before the genesis date.
            prior_days_window=7,
            batch_size=16,
            validation_size=1,
        ),
        instance=types.InstanceConfig(
            lookback_window=5,
            n_year_lookback=1,
        ),
        metrics=types.MetricsConfig(
            scorers=["neg_mean_absolute_error", "neg_mean_squared_error"],
        ),
        forecast=types.ForecastConfig(
            n_days=4,
        )
    )
    location_query = "Atlanta, GA US"

    logger.info(f"training model with config:\n{json.dumps(config, indent=4, default=str)}")

    get_training_instance = cache.cache_instance(
        data.get_training_instance, cache_dir="./.cache/training_data", **asdict(config.instance)
    )
    update_model_fn = cache.cache_model(
        update_model,
        eval_model_fn=partial(evaluate_model, config.metrics.scorers),
        cache_dir="./.cache/models",
        config=config,
    )

    model = None
    model_id = None

    # train model with specified amount of prior days knowledge
    date_now = datetime.datetime.now().date()
    genesis_to_now = (date_now - config.model.genesis_date).days + 1

    for i, n_days in enumerate(
        itertools.chain(
            # prior days knowledge lookback from the genesis date
            (-x for x in reversed(range(config.model.prior_days_window + 1))),
            # update model from genesis date to current date
            range(1, genesis_to_now),
        )
    ):
        if model is None:
            model = SGDRegressor(
                penalty="l1",
                alpha=1.0,
                random_state=567,
                warm_start=True,
                early_stopping=False,
            )
        target_date = config.model.genesis_date + datetime.timedelta(days=n_days)
        logger.info(f"[batch {i}] getting training/validation batches for target date {target_date}")
        training_batch, validation_batch = load_data(get_training_instance, location_query, target_date, config)
        if stop_training(training_batch):
            break
        model, model_id = update_model_fn(model, training_batch, validation_batch, target_date)

    # produce an n-day forecast
    predictions, forecast_dates = [], []
    for i, n_days in enumerate(range(config.forecast.n_days + 1)):
        forecast_date = date_now + datetime.timedelta(days=n_days)
        training_batch, _ = load_data(get_training_instance, location_query, forecast_date, config)
        pred = forecast_today(model, prepare_forecast_batch(training_batch, predictions))
        logger.info(f"[forecasting {i}] for target_date: {forecast_date}, prediction: {pred}")
        # most recent forecasts first
        predictions.insert(0, pred)
        forecast_dates.insert(0, forecast_date)
        logger.info("=" * 50)

    assert model is not None, f"model {model} cannot be None"
    assert model_id is not None, f"model_id {model_id} cannot be None"

    forecast = types.Forecast(
        created_at=date_now,
        model_id=model_id,
        predictions=[types.Prediction(value=p, date=d) for p, d in zip(predictions, forecast_dates)]
    )
    forecast_dir = Path(".cache") / "forecasts" / date_now.strftime("%Y%m%d") / model_id
    forecast_dir.mkdir(exist_ok=True, parents=True)
    with (forecast_dir / "forecasts.json").open("w") as f:
        json.dump(forecast, f, default=str, indent=4)
