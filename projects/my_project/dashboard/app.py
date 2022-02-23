import os
from argparse import ArgumentParser
from pathlib import Path

import streamlit as st

import pip

package_names=['flytekit','sklearn'] #packages to install
pip.main(['install'] + package_names + ['--upgrade']) 

from flytekit.remote import FlyteRemote
from flytekit.models import filters
from flytekit.models.admin.common import Sort

from sklearn.datasets import load_digits


PROJECT_NAME = "flytelab-my_project".replace("_", "-")
WORKFLOW_NAME = "my_project.workflows.main"


parser = ArgumentParser()
parser.add_argument("--remote", action="store_true")
args = parser.parse_args()

backend = os.getenv("FLYTE_BACKEND", 'remote' if args.remote else 'sandbox')

# configuration for accessing a Flyte cluster backend
remote = FlyteRemote.from_config(
    default_project=PROJECT_NAME,
    default_domain="development",
    config_file_path=Path(__file__).parent / f"{backend}.config",
)

# get the latest workflow execution
[latest_execution, *_], _ = remote.client.list_executions_paginated(
    PROJECT_NAME,
    "development",
    limit=1,
    filters=[
        filters.Equal("launch_plan.name", WORKFLOW_NAME),
        filters.Equal("phase", "SUCCEEDED"),
    ],
    sort_by=Sort.from_python_std("desc(execution_created_at)"),
)

wf_execution = remote.fetch_workflow_execution(name=latest_execution.id.name)
remote.sync(wf_execution, sync_nodes=False)
model = wf_execution.outputs["o0"]
print(model)


############
# App Code #
############

data = load_digits(as_frame=True)

st.write("# Flytelab: my_project")
st.write("### wfew")
st.write(f"Model: `{model}`")

st.write("Use the slider below to select a sample for prediction")

sample_index = st.slider(
    "Sample Number",
    min_value=0,
    max_value=data.frame.shape[0] - 1,
    value=0,
    step=1,
)

st.image(data.images[sample_index], clamp=True, width=300)
st.write(f"Ground Truth: {data.target[sample_index]}")
st.write(f"Prediction: {model.predict(data.frame[data.feature_names].loc[[sample_index]])[0]}")
