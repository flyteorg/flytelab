import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from flytekit import task, workflow


@task
def get_dataset() -> pd.DataFrame:
    return load_digits(as_frame=True).frame


@task
def train_model(dataset: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression()
    features, target = dataset[[x for x in dataset if x != "target"]], dataset["target"]
    return model.fit(features, target)


@workflow
def main() -> LogisticRegression:
    return train_model(dataset=get_dataset())


if __name__ == "__main__":
    print(f"trained model: {main()}")
