import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import pandas as pd
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
    "seattle": "seattle_weather_forecast_v2",
    "atlanta": "atlanta_weather_forecast_v2",
    "hyderabad": "hyderabad_weather_forecast_v2",
    "mumbai": "mumbai_weather_forecast_v2",
    "taipei": "taipei_weather_forecast_v2",
    "appleton": "appleton_weather_forecast_v2",
    "dharamshala": "dharamshala_weather_forecast_v2",
    "fremont": "fremont_weather_forecast_v2",
}


CITY_LABEL_MAP = {
    "atlanta": "Atlanta, GA USA",
    "seattle": "Seattle, WA USA",
    "hyderabad": "Hyderabad, Telangana India",
    "mumbai": "Mumbai, MH India",
    "taipei": "Taipei, Taiwan",
    "appleton": "Green Bay, WI USA",
    "dharamshala": "Dharamsala, HP India",
    "fremont": "Fremont, CA USA",
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

_, _, col, *_ = st.columns(5)
with col:
    st.image(LOGO, width=100)
st.title("Flytelab: Weather Forecasts ⛈☀️☔️")

"""
This app displays the weather forecasts produced by a model
that was trained using [flyte](https://flyte.org/). For more information
see the [flytelab weather forecasting project](https://github.com/flyteorg/flytelab/tree/main/projects/weather_forecasting).
"""

selected_city = st.selectbox(
    "Select a City",
    options=[
        "atlanta",
        "seattle",
        "hyderabad",
        "mumbai",
        "taipei",
        "appleton",
        "dharamshala",
        "fremont",
    ],
    format_func=lambda x: CITY_LABEL_MAP[x]
)

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

with st.expander("Model Metadata"):
    st.markdown(f"""
    ```
    model_id: {forecast.model_id}
    created_at: {forecast.created_at}
    training exp-weighted-mae: {scores.train_exp_mae}
    validation exp-weighted-mae: {scores.valid_exp_mae}
    ```
    """)

st.markdown(f"""
## {CITY_LABEL_MAP[selected_city]}

Air Temperature and Dew Temperature Forecast (°C)
""")

air_temp = []
dew_temp = []
datetime_index = []
for p in forecast.predictions:
    date = p.date.replace(tzinfo=None)
    if date < pd.Timestamp.now().floor("D").to_pydatetime():
        continue
    air_temp.append(p.air_temp)
    dew_temp.append(p.dew_temp)
    datetime_index.append(date)

data = pd.DataFrame(
    {"air_temp": air_temp, "dew_temp": dew_temp},
    index=datetime_index
)

st.line_chart(data)

st.markdown(f"""
Predictions powered by [flyte](https://flyte.org/)
""")
