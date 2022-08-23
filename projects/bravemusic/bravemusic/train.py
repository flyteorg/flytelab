import json
import typing
import warnings
import numpy as np
from tensorflow import keras
from dataclasses import dataclass
from preprocess import preprocess
from datasource import GTZAN_ZIP_FILE_PATH
from dataclasses_json import dataclass_json
from flytekit.types.directory import FlyteDirectory
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")
MODELSAVE = [typing.TypeVar("str")]
model_file = typing.NamedTuple("Model", model=FlyteDirectory[MODELSAVE])


@dataclass_json
@dataclass
class Hyperparameters(object):
    batch_size: int = 32
    metrics: str = "accuracy"
    loss = ("sparse_categorical_crossentropy",)
    epochs: int = 30
    learning_rate: float = 0.0001


def train(
    data: dict,
    hp: Hyperparameters
) -> model_file:
    # with open("data.json", "r") as fp:
    #     data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    optimiser = keras.optimizers.Adam(learning_rate=hp.learning_rate)
    model.compile(
        optimizer=optimiser,
        loss=hp.loss,
        metrics=[hp.metrics],
    )
    # train model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=hp.batch_size,
        epochs=hp.epochs,
    )

    Dir = "model"
    model.save(Dir)
    return model


if __name__ == '__main__':
    data = preprocess(dataset_path=GTZAN_ZIP_FILE_PATH)
    model = train(
        data=data,
        hp=Hyperparameters(epochs=1)
    )
