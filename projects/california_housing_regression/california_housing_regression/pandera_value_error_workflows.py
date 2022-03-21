from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from flytekit import task, workflow

from california_housing_regression.pandera_workflows import (
    Hyperparameters,
    DatasetSplits,
    TrainingResult,
    summarize_dataset,
    train_model,
    evaluate_model,
)


@task
def get_dataset(test_size: float, random_state: int) -> DatasetSplits:
    dataset = fetch_california_housing(as_frame=True).frame
    training_set, test_set = train_test_split(dataset, test_size=test_size, random_state=random_state)
    # corrupt the test set
    test_set.loc[:, "Latitude"].iloc[:5] = -1000
    return training_set, test_set


@workflow
def main(hyperparameters: Hyperparameters, test_size: float = 0.2, random_state: int = 43) -> TrainingResult:
    train_dataset, test_dataset = get_dataset(test_size=test_size, random_state=random_state)

    summarize_dataset(dataset=train_dataset)
    summarize_dataset(dataset=test_dataset)

    model = train_model(dataset=train_dataset, hyperparameters=hyperparameters)
    train_mse = evaluate_model(dataset=train_dataset, model=model)
    test_mse = evaluate_model(dataset=test_dataset, model=model)

    return model, train_mse, test_mse


if __name__ == "__main__":
    print(f"trained model: {main(hyperparameters=Hyperparameters(alpha=10.0))}")
