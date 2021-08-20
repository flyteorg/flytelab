import json
import datetime
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from typing import List, Optional, TypedDict, Union


DateType = Union[datetime.date, datetime.datetime]


def date_field_config():
    return config(
        encoder=lambda x: (
            datetime.date.isoformat(x)
            if isinstance(x, datetime.date)
            else datetime.datetime.isoformat(x)
        ),
        decoder=lambda x: (
            datetime.date.fromisoformat(x)
            if isinstance(x, datetime.date)
            else datetime.datetime.fromisoformat(x)
        ),
    )


def features_field_config():
    return config(
        encoder=lambda x: json.dumps([float(i) for i in x]),
        decoder=lambda x: json.loads(x),
    )


@dataclass_json
@dataclass
class ModelConfig:
    genesis_date: datetime.datetime = field(metadata=date_field_config())
    prior_days_window: int
    batch_size: int
    validation_size: int


@dataclass_json
@dataclass
class InstanceConfig:
    lookback_window: int
    n_year_lookback: int


@dataclass_json
@dataclass
class MetricsConfig:
    scorers: List[str]


@dataclass_json
@dataclass
class ForecastConfig:
    n_days: int


@dataclass_json
@dataclass
class Config:
    model: ModelConfig
    instance: InstanceConfig
    metrics: MetricsConfig
    forecast: ForecastConfig


Metrics = TypedDict(
    "Metrics",
    name=str,
    train=float,
    train_size=int,
    validation=Optional[float],
    validation_size=int,
)


@dataclass_json
@dataclass
class Prediction:
    value: Optional[float]
    date: datetime.datetime = field(metadata=date_field_config())
    error: Optional[str] = None


@dataclass_json
@dataclass
class Forecast:
    created_at: datetime.datetime = field(metadata=date_field_config())
    model_id: str
    predictions: List[Prediction]
