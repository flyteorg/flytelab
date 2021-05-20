import os

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
client = SynchronousFlyteClient("demo.nuclyde.io")

_, _, col, *_ = st.beta_columns(5)
with col:
    st.image(LOGO, width=100)
st.title("Flytelab: Weather Forecasts ⛈☀️☔️")

"""
This app displays the weather forecasts produced by a model
that was trained using [flyte](https://flyte.org/). For more information
see the [flytelab weather forecasting project](https://github.com/flyteorg/flytelab/tree/main/projects/weather_forecasting).
"""

executions, _ = client.list_executions_paginated(
    "flytelab",
    "development",
    limit=1,
    filters=[
        filters.Equal("workflow.name", "app.workflow.run_pipeline"),
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

st.markdown("""
## Atlanta, GA USA
---
""")
for prediction in forecast.predictions[::-1]:
    st.markdown(f"### {prediction.date.date().strftime('%m/%d/%Y')}")
    st.markdown(f"🌡 **Mean Temperature**: {prediction.value:0.02f} °C")
    st.markdown("---")
