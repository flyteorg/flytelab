import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
import mlflow
from fastapi import FastAPI, File, UploadFile
from pip import main
import tensorflow as tf
import librosa
import math

# TODO
# To accept Audio file sending from streamlit


# Initiate app instance
app = FastAPI(title="Brave Hyena", version="1.0", description="Trying Locally")

# in deployment we use remote.fetch_workflow_execution to get the model
model = tf.keras.models.load_model("< Copy paste the flyteworkflow output url>")
genre = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}


# Api root or home endpoint
@app.get("/")
@app.get("/home")
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    return {"message": "Looks Good"}


data = {"mfcc": []}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Extract data in correct order
    hop_length = 512
    num_segments = 10
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    audio, sample_rate = librosa.load(file.file, 22050)
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(
            audio[start:finish], sample_rate, n_mfcc=13, n_fft=2048, hop_length=512
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
    prediction = np.argmax(predict_x, axis=1)

    return {"Genre": genre[round(prediction.mean())]}
    return item


@app.get("/")
async def root():
    def main():
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
