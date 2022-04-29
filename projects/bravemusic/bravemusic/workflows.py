import typing
from dataclasses import dataclass
import numpy as np
from flytekit.types.directory import FlyteDirectory
import pandas as pd
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from typing import NamedTuple, Tuple
import flytekit
import joblib
import os
import json
import math
import librosa
import warnings
warnings.filterwarnings("ignore")
import os
import urllib
import tarfile

data = {
    "mapping": [],
    "labels": [],
    "mfcc": []
}

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def preprocess(
    dataset_path: str, json_path: str, num_mfcc: int, n_fft: int, hop_length: int, num_segments: int
    ):
    
    #Downloading data from web
    testfile = urllib.request.URLopener()
    testfile.retrieve("http://opihi.cs.uvic.ca/sound/genres.tar.gz", "genres.tar.gz")
   
    #open file
    file = tarfile.open("genres.tar.gz")
    file.extractall(".../data")
    file.close()
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

		# load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

MODELSAVE = [typing.TypeVar("str")]
model_file = typing.NamedTuple("Model", model=FlyteDirectory[MODELSAVE])
workflow_outputs = typing.NamedTuple(
    "WorkflowOutputs",  model=FlyteDirectory[MODELSAVE])

@dataclass_json
@dataclass
class Hyperparameters(object):
    batch_size: int = 32
    metrics: str = "accuracy"
    loss='sparse_categorical_crossentropy',
    epochs: int = 30
    learning_rate: float = 0.0001

@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def train(hp: Hyperparameters, json_path: str,
) -> model_file:

    with open(json_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    optimiser = keras.optimizers.Adam(learning_rate=hp.learning_rate)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=hp.batch_size, epochs=hp.epochs)
    
    #fname = "model.json 
    Dir ="model" 
    model.save(Dir)
    #json.dump(model, fname)
    return (Dir,)

@workflow
def flyteworkflow(
    dataset_path: str = ".../data/genres",
    json_path: str = "data.json",
    num_mfcc: int = 13, 
    n_fft: int = 2048, 
    hop_length: int = 512, 
    num_segments: int = 10,
)-> workflow_outputs:

    preprocess(dataset_path=dataset_path,json_path=json_path,num_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length, num_segments=num_segments)
    model = train(hp=Hyperparameters(epochs=30),json_path=json_path, )

    return model.model,

if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print(flyteworkflow())




