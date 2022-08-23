import os
from argparse import ArgumentParser
from pathlib import Path

import streamlit as st

from flytekit.remote import FlyteRemote
from flytekit.models import filters
from flytekit.models.admin.common import Sort

from sklearn.datasets import load_digits
from sqlite3 import DatabaseError
from itsdangerous import json
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from pydub import AudioSegment
import librosa
import math
import tensorflow as tf


PROJECT_NAME = "flytelab-final".replace("_", "-")
WORKFLOW_NAME = "final.workflows.main"


parser = ArgumentParser()
parser.add_argument("--remote", action="store_true")
args = parser.parse_args()

backend = os.getenv("FLYTE_BACKEND", "remote" if args.remote else "sandbox")

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
modelurl = wf_execution.outputs["o0"]
print(model)


############
# App Code #
############


with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Project Design", "Meet The Team"],  # required
        icons=["house", "diagram-2", "people"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )


if selected == "Home":
    st.markdown(
        """<h2 style='text-align: center; color: #FF0080;font-size:60px;margin-top:-50px;'>Music Genre Classification</h2>""",
        unsafe_allow_html=True,
    )
    # in deployment we use remote.fetch_workflow_execution to get the model url
    model = tf.keras.models.load_model(modelurl)
    genre = {
        0: "Blues",
        1: "Classical",
        2: "Country",
        3: "Disco",
        4: "Hiphop",
        5: "Jazz",
        6: "Metal",
        7: "Pop",
        8: "Reggae",
        9: "Rock",
    }

    global type
    UploadAudio = st.file_uploader("Upload Music To Classify", type=["wav", "mp3"])
    st.markdown("""<h3 style='color:#FF0080;'> Play: </h3>""", unsafe_allow_html=True)
    st.audio(UploadAudio)
    hop_length = 512
    num_segments = 10
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    data = {"mfcc": []}

    if st.button("Predict"):
        if UploadAudio is not None:
            type = UploadAudio.type
            if type == "audio/mpeg":
                UploadAudio = AudioSegment.from_mp3(UploadAudio)
                UploadAudio.export("file.wav", format="wav")
                samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
                num_mfcc_vectors_per_segment = math.ceil(
                    samples_per_segment / hop_length
                )
                audio, sample_rate = librosa.load(UploadAudio, 22050)
                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    mfcc = librosa.feature.mfcc(
                        audio[start:finish],
                        sample_rate,
                        n_mfcc=13,
                        n_fft=2048,
                        hop_length=512,
                    )
                    mfcc = mfcc.T
                    break

                data["mfcc"].append(mfcc.tolist()) if len(
                    mfcc
                ) == num_mfcc_vectors_per_segment else print(
                    "It's not the same as the Trained data"
                )

                test = np.array(data["mfcc"])
                predict_x = model.predict(test)
                predictions = np.argmax(predict_x, axis=1)
                prediction = genre[round(predictions.mean())]

                st.markdown(
                    f"""<h1 style='color:#FF0080;'>You're Listening to : <span style='color:#151E3D;'>{prediction}</span></h1>""",
                    unsafe_allow_html=True,
                )

            else:
                samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
                num_mfcc_vectors_per_segment = math.ceil(
                    samples_per_segment / hop_length
                )
                audio, sample_rate = librosa.load(UploadAudio, 22050)
                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    mfcc = librosa.feature.mfcc(
                        audio[start:finish],
                        sample_rate,
                        n_mfcc=13,
                        n_fft=2048,
                        hop_length=512,
                    )
                    mfcc = mfcc.T
                    break

                data["mfcc"].append(mfcc.tolist()) if len(
                    mfcc
                ) == num_mfcc_vectors_per_segment else print(
                    "It's not the same as the Trained data"
                )

                test = np.array(data["mfcc"])
                predict_x = model.predict(test)
                predictions = np.argmax(predict_x, axis=1)
                prediction = genre[round(predictions.mean())]

                st.markdown(
                    f"""<h1 style='color:#FF0080;'>You're Listening to : <span style='color:#151E3D;'>{prediction}</span></h1>""",
                    unsafe_allow_html=True,
                )
                # st.success(f"You're Listening to: {genre[round(prediction.mean())]}")


if selected == "Project Design":
    st.markdown(
        """<h2 style='text-align: center; color: purple;font-size:60px;margin-top:-50px;'>Our Project Holistic View</h2>""",
        unsafe_allow_html=True,
    )
if selected == "Meet The Team":
    st.markdown(
        """<h2 style='text-align: center; color: purple;font-size:60px;margin-top:-50px;'>Meet Our Amazing Team</h2>""",
        unsafe_allow_html=True,
    )
