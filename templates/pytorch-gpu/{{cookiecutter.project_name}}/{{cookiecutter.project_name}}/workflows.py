import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits

from flytekit import task, workflow, Resources


dataset_resources = Resources(cpu="1", mem="1Gi", storage="1Gi")

# This conditional is used at deployment time to determine whether the
# task uses CPUs or GPUs. The "FLYTE_SANDBOX" environment variable is
# automatically set by the `deploy.py` script when serializing tasks/workflows
training_resources = (
    Resources(cpu="1", mem="1Gi", storage="1Gi")
    if int(os.getenv("FLYTE_SANDBOX", "0"))
    else Resources(gpu="1", mem="4Gi", storage="4Gi")
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        return F.log_softmax(self.layer3(x))


@task(requests=dataset_resources, limits=dataset_resources)
def get_dataset() -> pd.DataFrame:
    return load_digits(as_frame=True).frame


def dataset_iterator(features, target, n_batches: int):
    for X, y in zip(np.array_split(features, n_batches), np.array_split(target, n_batches)):
        yield (
            torch.from_numpy(X.values).float().to(DEVICE),
            torch.from_numpy(y.values).long().to(DEVICE)
        )


@task(requests=training_resources, limits=training_resources)
def train_model(
    dataset: pd.DataFrame,
    hidden_dim: int,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Model:
    features, target = dataset[[x for x in dataset if x != "target"]], dataset["target"]

    # define the model
    n_classes = target.nunique()
    model = Model(features.shape[1], hidden_dim, n_classes).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=learning_rate)

    # iterate through n_epochs and n_batches of the training data
    n_batches = int(features.shape[0] / batch_size)
    for epoch in range(1, n_epochs + 1):
        for batch, (X, y) in enumerate(dataset_iterator(features, target, n_batches), 1):

            opt.zero_grad()
            y_hat = model(X)
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            opt.step()

            accuracy = (y_hat.argmax(1) == y).float().mean()

            print(
                f"epoch={epoch:02d}: "
                f"batch {batch:02d}/{n_batches} - "
                f"loss={loss.item():0.04f}; "
                f"accuracy={accuracy:0.04f}"
            )

    return model.to("cpu")


@workflow
def main(
    hidden_dim: int = 300,
    n_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> Model:
    return train_model(
        dataset=get_dataset(),
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    print(f"trained model: {main()}")
