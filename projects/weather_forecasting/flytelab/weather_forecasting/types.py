import datetime
from typing import List, Optional, TypedDict


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

ForecastConfig = TypedDict(
    "ForecastConfig",
    n_days=int,
)

Config = TypedDict(
    "Config",
    model=ModelConfig,
    instance=InstanceConfig,
    metrics=MetricsConfig,
    forecast=ForecastConfig,
)

Metrics = TypedDict(
    "Metrics",
    name=str,
    train=float,
    train_size=int,
    validation=Optional[float],
    validation_size=int,
)

Prediction = TypedDict(
    "Prediction",
    value=float,
    date=datetime.datetime
)

Forecast = TypedDict(
    "Forecast",
    created_at=datetime.datetime,
    model_id=str,
    predictions=List[Prediction],
)
