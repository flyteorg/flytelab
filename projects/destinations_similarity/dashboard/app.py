import os
from argparse import ArgumentParser
from pathlib import Path

import streamlit as st
import logging
import pandas as pd
import requests
from flytekit.remote import FlyteRemote
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from PIL import Image
from sklearn.datasets import load_digits


PROJECT_NAME = "vamos-dalhe"
WORKFLOW_NAME = "destinations_similarity.workflows.main"

# Logging config
LOGGER = logging.getLogger(__name__)

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

def retrieve_dataset_from_remote(url: str) -> pd.DataFrame:
    """Retrieve the dataset from a remote URL.

    Args:
        url (str): Remote address of the dataset, a Parquet file.

    Returns:
        pd.DataFrame: DataFrame with the dataset.
    """
    dataset_parquet = requests.get(url, timeout=30)
    dataset_df = pd.read_parquet(dataset_parquet)
    LOGGER.info("Retrieved dataset from %s.", url)
    return dataset_df

df = retrieve_dataset_from_remote("https://storage.googleapis.com/dsc-public-info/datasets/flytelab_dataset.parquet")

st.write("# Flytelab: destinations_similarity")
#st.write("## Hurb project to the Flyte Hackathon")
st.write('## Kinder is an adventurous dog who loves to travel! He enjoys specially nature places: beachs, waterfalls, trails and more, which Brazil is not missing. He wants experiences in other cities but he doesnt know where. So he is now asking, **where should I go next**?')

beach_kinder = Image.open('beach_kinder.jpg')

st.image(beach_kinder, caption='Kinder in love with the beach')

st.write("Help Kinder by selecting a city you like in Brazil below so we can recommend similar places he will most certainly enjoy!")
desired_city = st.selectbox('I like:', df['city'].unique())
n_citys = st.slider('How many recommendations do you want?', 1, 30, 5)

st.write("## So, where next?")
st.write("Kinder should go to: " + next_destination(df,n_citys,768,desired_city))

st.write("Hope you enjoy the recommendation! See you on your next trip.")

kinder = Image.open('kinder.jpg')

st.image(kinder, caption='The marvelous Kinder')