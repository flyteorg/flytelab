import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import streamlit as st
from dataclasses_json import dataclass_json

from flytekit.remote import FlyteRemote
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from google.protobuf.json_format import MessageToJson


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


LOGO = "https://docs.flyte.org/en/latest/_static/flyte_circle_gradient_1_4x4.png"

LAUNCH_PLAN_MAP = {
    "seattle_wa_usa": "seattle_weather_forecast_v2",
    "atlanta_ga_usa": "atlanta_weather_forecast_v2",
    "hyderabad_tn_in": "hyderabad_weather_forecast_v2",
}


remote = FlyteRemote.from_config(
    default_project="flytelab",
    default_domain="development",
    config_file_path=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "flyte.config"
    )
)

st.set_page_config(
    page_title="flytelab - weather forecasts",
    page_icon=LOGO,
)

_, _, col, *_ = st.beta_columns(5)
with col:
    st.image(LOGO, width=100)
st.title("Flytelab: Weather Forecasts ⛈☀️☔️")

"""
This app displays the weather forecasts produced by a model
that was trained using [flyte](https://flyte.org/). For more information
see the [flytelab weather forecasting project](https://github.com/flyteorg/flytelab/tree/main/projects/weather_forecasting).
"""

city_label_map = {
    "atlanta_ga_usa": "Atlanta, GA USA",
    "seattle_wa_usa": "Seattle, WA USA",
    "hyderabad_ga_usa": "Hyderabad, Telangana India",
}

selected_city = st.selectbox(
    "Select a City",
    options=["atlanta_ga_usa", "seattle_wa_usa", "hyderabad_ga_usa"],
    format_func=lambda x: city_label_map[x]
)

selected_city = "atlanta_ga_usa"

executions, _ = remote.client.list_executions_paginated(
    "flytelab",
    "development",
    limit=1,
    filters=[
        filters.Equal("launch_plan.name", LAUNCH_PLAN_MAP[selected_city]),
        filters.Equal("phase", "SUCCEEDED"),
    ],
    sort_by=Sort.from_python_std("desc(execution_created_at)"),
)

wf_execution_output = remote.client.get_execution_data(executions[0].id)
literals = wf_execution_output.full_outputs.literals
forecast = Forecast.from_json(MessageToJson(literals["forecast"].scalar.generic))
scores = Scores.from_json(MessageToJson(literals["scores"].scalar.generic))

with st.beta_expander("Model Metadata"):
    st.markdown(f"""
    ```
    model_id: {forecast.model_id}
    created_at: {forecast.created_at}
    ```
    """)

st.markdown(f"""
## {city_label_map[selected_city]}
---
""")

for prediction in forecast.predictions[::-1]:
    if prediction.date.replace(tzinfo=None) < datetime.now():
        continue
    st.markdown(f"### {prediction.date.strftime('%m/%d/%Y %H:%M:%S')}")
    st.markdown(f"🌡 **Air Temperature**: {prediction.air_temp:0.02f} °C")
    st.markdown("---")
