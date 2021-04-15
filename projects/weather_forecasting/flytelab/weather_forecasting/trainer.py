import datetime
import json
from functools import lru_cache, partial, wraps
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, List, Optional, TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import SCORERS

from flytelab.weather_forecasting import data


ModelConfig = TypedDict(
    "ModelConfig",
    genesis_date=datetime.date,
    prior_days_window=int,
    batch_size=int,
    validation_size=int,
)

InstanceConfig = TypedDict(
    "InstanceConfig",
    lookback_window=int,
    n_year_lookback=int,
)

MetricsConfig = TypedDict(
    "MetricConfig",
    scorers=List[str],
)

Config = TypedDict(
    "Config",
    model=ModelConfig,
    instance=InstanceConfig,
    metrics=MetricsConfig,
)

Metrics = TypedDict(
    "Metrics",
    name=str,
    train=float,
    train_size=int,
    validation=float,
    validation_size=int,
)


@lru_cache
def create_hash_id(*data):
    _hash = md5()
    for d in data:
        _hash.update(d)
    return _hash.hexdigest()[:7]


@lru_cache
def create_id(target_date, *data):
    return f"{target_date.strftime(r'%Y%m%d')}_{create_hash_id(*data)}"


def cache_instance(get_instance_fn=None, *, cache_dir, **instance_config):
    """Decorator to automatically cache training instances."""

    if get_instance_fn is None:
        return partial(cache_instance, cache_dir=cache_dir)

    cache_data_id = create_hash_id(json.dumps(instance_config).encode())
    cache_dir = Path(cache_dir) / cache_data_id

    if not Path(cache_dir).exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    config_path = cache_dir / "config.json"
    if not config_path.exists():
        with config_path.open("w") as f:
            json.dump(instance_config, f, indent=4, default=str)

    @lru_cache
    def get_instance(instance_id, target_date):
        cache_path = cache_dir / f"{instance_id}.json"
        if cache_path.exists():
            with cache_path.open("r") as f:
                print(f"getting training instance {instance_id} from cache dir {cache_dir}")
                return data.TrainingInstance(**json.load(f))

        instance = get_instance_fn(location_query, target_date, instance_id=instance_id, **instance_config)
        with cache_path.open("w") as f:
            json.dump({
                "features": instance.features,
                "target": instance.target,
                "id": instance.id,
            }, f)

        return instance

    @wraps(get_instance_fn)
    def wrapped_instance_fn(
        location_query: str,
        target_date: data.DateType,
    ):
        return get_instance(
            create_id(
                target_date,
                location_query.encode(),
                target_date.isoformat().encode(),
                json.dumps(instance_config).encode(),
            ),
            target_date,
        )

    return wrapped_instance_fn


def cache_model(
    update_model_fn: Callable[[BaseEstimator, List[data.TrainingInstance]], BaseEstimator] = None,
    *,
    eval_model_fn: Callable[
        [BaseEstimator, List[data.TrainingInstance], List[data.TrainingInstance]],
        List[Metrics],
    ],
    cache_dir,
    **config,
):
    """Decorator to automatically cache model updates."""
    if update_model_fn is None:
        return partial(cache_model, cache_dir=cache_dir)
    
    # consists of "{genesis_date}_{hash_of_config}"
    model_id_data = [str(config[key]).encode() for key in ["model", "instance"]]
    model_dir_id = create_id(config["model"]["genesis_date"], *model_id_data)
    cache_model_dir = Path(cache_dir) / model_dir_id

    # path for root model (no training data)
    root_model_id = f"00000000_{create_hash_id(*model_id_data)}"
    root_model_dir = cache_model_dir / root_model_id
    root_model_path = root_model_dir / "model.joblib"

    for path in (cache_model_dir, root_model_dir):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    config_path = cache_model_dir / "config.json"
    if not config_path.exists():
        with config_path.open("w") as f:
            json.dump(config, f, indent=4, default=str)

    @wraps(update_model_fn)
    def wrapped_update_model_fn(
        model: BaseEstimator,
        training_batch: List[data.TrainingInstance],
        validation_batch: List[data.TrainingInstance],
        target_date: datetime.date,
    ):
        if not root_model_path.exists():
            print(f"writing root model to {root_model_path}")
            joblib.dump(model, root_model_path, compress=True)

        model_update_id = create_id(
            target_date, joblib.hash(model).encode(), *[instance.id.encode() for instance in training_batch]
        )
        model_update_dir = cache_model_dir / model_update_id
        if not model_update_dir.exists():
            model_update_dir.mkdir(parents=True)

        model_update_path = model_update_dir / "model.joblib"
        model_metrics_path = model_update_dir / "metrics.json"

        if all(p.exists() for p in (model_update_path, model_metrics_path)):
            print(f"reading model from cache {model_update_path}")
            with model_metrics_path.open() as f:
                metrics = json.load(f)
            for metric in metrics:
                print(f"[metric] {metric}")
            model = joblib.load(model_update_path)
            return model

        print(f"updating model and writing to {model_update_path}")
        updated_model = update_model_fn(model, training_batch)
        joblib.dump(updated_model, model_update_path, compress=True)

        metrics = eval_model_fn(updated_model, training_batch, validation_batch)
        with model_metrics_path.open("w") as f:
            for metric in metrics:
                print(f"[metric] {metric}")
            json.dump(metrics, f, indent=4, default=str)
        return updated_model

    return wrapped_update_model_fn


def load_data(get_training_instance, location_query, target_date, config):
    training_batch = [
        get_training_instance(location_query, target_date - datetime.timedelta(days=i))
        for i in range(config["model"]["batch_size"])
    ]
    validation_batch = [
        instance for instance in (
            get_training_instance(location_query, target_date + datetime.timedelta(days=i + 1))
            for i in range(config["model"]["validation_size"])
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
        try:
            metrics.append(
                Metrics(
                    name=scorer,
                    train=scoring_fn(model, train_features, train_target),
                    train_size=len(training_batch),
                    validation=scoring_fn(model, validation_features, validation_target),
                    validation_size=len(validation_batch),
                )
            )
        except:
            import ipdb; ipdb.set_trace()
    return metrics


if __name__ == "__main__":

    config = Config(
        model=ModelConfig(
            genesis_date=(datetime.datetime.now() - datetime.timedelta(days=3)).date(),
            # lookback window for pre-training the model n number of days before the genesis date.
            prior_days_window=3,
            batch_size=16,
            validation_size=4,
        ),
        instance=InstanceConfig(
            lookback_window=30,
            n_year_lookback=1,
        ),
        metrics=MetricsConfig(
            scorers=["neg_mean_absolute_error", "neg_mean_squared_error"],
        )
    )
    location_query = "Atlanta, GA US"

    print(f"training model with config:\n{json.dumps(config, indent=4, default=str)}")

    get_training_instance = cache_instance(
        data.get_training_instance, cache_dir="./.cache/training_data", **config["instance"]
    )
    update_model_fn = cache_model(
        update_model,
        eval_model_fn=partial(evaluate_model, config["metrics"]["scorers"]),
        cache_dir="./.cache/models",
        **config
    )
    
    model = None

    # train model with specified amount of prior days knowledge
    for i, n_prior_window in enumerate(reversed(range(config["model"]["prior_days_window"] + 1)), 1):
        if model is None:
            model = SGDRegressor(
                penalty="l1",
                alpha=1.0,
                random_state=567,
                warm_start=True,
                early_stopping=False,
            )
        target_date = config["model"]["genesis_date"] - datetime.timedelta(days=n_prior_window)
        print(f"[batch {i}] getting training/validation batches for target date {target_date}")
        training_batch, validation_batch = load_data(get_training_instance, location_query, target_date, config)
        if stop_training(training_batch):
            print(f"[stop training] target is None for target_date: {target_date}")
            break
        model = update_model_fn(model, training_batch, validation_batch, target_date)
        print("=" * 50)
