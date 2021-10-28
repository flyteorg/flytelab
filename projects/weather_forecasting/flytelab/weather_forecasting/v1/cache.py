import datetime
import logging
import joblib
import json
from dataclasses import asdict
from hashlib import md5
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator

from flytelab.weather_forecasting.v1 import data, types


logger = logging.getLogger(__file__)


@lru_cache
def create_hash_id(*data):
    _hash = md5()
    for d in data:
        _hash.update(d)
    return _hash.hexdigest()[:7]


@lru_cache
def create_id(date, *data) -> str:
    return f"{date.strftime(r'%Y%m%d')}_{create_hash_id(*data)}"


def create_model_id(
    genesis_date: datetime.date,
    update_date: datetime.date,
    model_config: types.ModelConfig,
    instance_config: types.InstanceConfig,
    instances: Optional[List[data.TrainingInstance]] = None,
) -> str:
    """Create a model identified by its genesis date, update date, and configuration
    """
    hash_data = [str(x).encode() for x in (model_config, instance_config)]
    if instances:
        hash_data = [*hash_data, *[instance.id.encode() for instance in instances]]
    return (
        f"{genesis_date.strftime(r'%Y%m%d')}_{update_date.strftime(r'%Y%m%d')}_{create_hash_id(*hash_data)}"
    )


def cache_instance(get_instance_fn: Callable[..., data.TrainingInstance] = None, *, cache_dir, **instance_config):
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
    def get_instance(instance_id, location_query, target_date):
        cache_path = cache_dir / f"{instance_id}.json"
        if cache_path.exists():
            with cache_path.open("r") as f:
                logger.info(f"getting training instance {instance_id} from cache dir {cache_dir}")
                return data.TrainingInstance(**json.load(f))

        instance = get_instance_fn(location_query, target_date, instance_id=instance_id, **instance_config)
        if not pd.isna(instance.target) and instance.target_is_complete:
            with cache_path.open("w") as f:
                json.dump(asdict(instance), f, default=str)

        return instance

    @wraps(get_instance_fn)
    def wrapped_instance_fn(
        location_query: str,
        target_date: types.DateType,
    ):
        return get_instance(
            create_id(
                target_date,
                location_query.encode(),
                target_date.isoformat().encode(),
                json.dumps(instance_config).encode(),
            ),
            location_query,
            target_date,
        )

    return wrapped_instance_fn


def cache_model(
    update_model_fn: Callable[[BaseEstimator, List[data.TrainingInstance]], BaseEstimator] = None,
    *,
    eval_model_fn: Callable[
        [BaseEstimator, List[data.TrainingInstance], List[data.TrainingInstance]],
        List[types.Metrics],
    ],
    cache_dir: str,
    config: types.Config,
):
    """Decorator to automatically cache model updates."""
    if update_model_fn is None:
        return partial(cache_model, cache_dir=cache_dir)

    # consists of "{genesis_date}_{hash_of_config}"
    model_id_data = [str(asdict(config)[key]).encode() for key in ["model", "instance"]]
    model_dir_id = create_id(config.model.genesis_date, *model_id_data)
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
            logger.info(f"writing root model to {root_model_path}")
            joblib.dump(model, root_model_path, compress=True)

        model_update_id = create_id(
            target_date, joblib.hash(model).encode(), *[instance.id.encode() for instance in training_batch]
        )
        model_id = f"{model_dir_id}/{model_update_id}"
        model_update_dir = cache_model_dir / model_update_id
        if not model_update_dir.exists():
            model_update_dir.mkdir(parents=True)

        model_update_path = model_update_dir / "model.joblib"
        model_metrics_path = model_update_dir / "metrics.json"

        if all(p.exists() for p in (model_update_path, model_metrics_path)):
            logger.info(f"reading model from cache {model_update_path}")
            with model_metrics_path.open() as f:
                metrics = json.load(f)
            for metric in metrics:
                log_metrics = {k: f"{v:0.04f}" if isinstance(v, float) else v for k, v in metric.items()}
                logger.info(f"[metric] {log_metrics}")
            model = joblib.load(model_update_path)
            return model, model_id

        logger.info(f"updating model and writing to {model_update_path}")
        updated_model = update_model_fn(model, training_batch)
        joblib.dump(updated_model, model_update_path, compress=True)

        metrics = eval_model_fn(updated_model, training_batch, validation_batch)
        for metric in metrics:
            log_metrics = {k: f"{v:0.04f}" if isinstance(v, float) else v for k, v in metric.items()}
            logger.info(f"[metric] {log_metrics}")
        # only cache metrics if number of instances in validation batch matches validation size
        if config.model.validation_size == len(validation_batch):
            with model_metrics_path.open("w") as f:
                json.dump(metrics, f, indent=4, default=str)

        return updated_model, model_id

    return wrapped_update_model_fn
