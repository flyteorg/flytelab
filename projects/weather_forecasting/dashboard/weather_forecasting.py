import os
import datetime

import streamlit as st

os.environ["FLYTE_INTERNAL_CONFIGURATION_PATH"] = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "flyte.config"
)


from flytekit.clients.friendly import SynchronousFlyteClient
from flytekit.control_plane.workflow_execution import FlyteWorkflowExecution
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytelab.weather_forecasting import types
from google.protobuf.json_format import MessageToJson


LOGO = "https://docs.flyte.org/en/latest/_static/flyte_circle_gradient_1_4x4.png"


st.set_page_config(
    page_title="flytelab - weather forecasts",
    page_icon=LOGO,
)
client = SynchronousFlyteClient("sandbox.uniondemo.run")

_, _, col, *_ = st.beta_columns(5)
with col:
    st.image(LOGO, width=100)
st.title("Flytelab: Weather Forecasts ⛈☀️☔️")

"""
This app displays the weather forecasts produced by a model
that was trained using [flyte](https://flyte.org/). For more information
see the [flytelab weather forecasting project](https://github.com/flyteorg/flytelab/tree/main/projects/weather_forecasting).
"""

launch_plan_map = {
    "seattle_wa_usa": "seattle_weather_forecast",
    "atlanta_ga_usa": "atlanta_weather_forecast",
}

city_label_map = {
    "atlanta_ga_usa": "Atlanta, GA USA",
    "seattle_wa_usa": "Seattle, WA USA",
}

selected_city = st.selectbox(
    "Select a City",
    options=["atlanta_ga_usa", "seattle_wa_usa"],
    format_func=lambda x: city_label_map[x]
)

executions, _ = client.list_executions_paginated(
    "flytelab",
    "development",
    limit=1,
    filters=[
        filters.Equal("launch_plan.name", launch_plan_map[selected_city]),
        filters.Equal("phase", "SUCCEEDED"),
    ],
    sort_by=Sort.from_python_std("desc(execution_created_at)"),
)

wf_execution = FlyteWorkflowExecution.fetch("flytelab", "development", executions[0].id.name)
forecast = types.Forecast.from_json(MessageToJson(wf_execution.outputs.literals['o0'].scalar.generic))

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
    if prediction.date.date() < datetime.datetime.now().date():
        continue
    st.markdown(f"### {prediction.date.date().strftime('%m/%d/%Y')}")
    st.markdown(f"🌡 **Mean Temperature**: {prediction.value:0.02f} °C")
    st.markdown("---")
