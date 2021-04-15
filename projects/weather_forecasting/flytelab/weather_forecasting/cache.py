import datetime
import joblib
import json
from hashlib import md5
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import List

from sklearn.base import BaseEstimator

from flytelab.weather_forecasting import data


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
