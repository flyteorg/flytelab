from dataclasses import asdict

from sklearn.linear_model import Ridge

from flytekit import task, workflow

from california_housing_regression.pandera_workflows import (
    TARGET,
    Hyperparameters,
    TrainingResult,
    get_dataset,
    summarize_dataset,
    train_model,
    evaluate_model,
)


@task
def train_model_type_error(dataset: dict, hyperparameters: Hyperparameters) -> Ridge:
    model = Ridge(**asdict(hyperparameters))
    return model.fit(dataset.drop(TARGET, axis="columns"), dataset[TARGET])

# TypeError: Cannot convert from scalar {
#   schema {
#     uri: "/tmp/flyte/20220319_170441/raw/f6608163de0159a39b9d21456bf4dc17"
#     type {
#       columns {name: "Latitude" type: FLOAT}
#       columns {name: "Longitude" type: FLOAT}
#       columns {name: "AveBedrms" type: FLOAT}
#       columns {name: "AveOccup" type: FLOAT}
#       columns {name: "AveRooms" type: FLOAT}
#       columns {name: "HouseAge" type: FLOAT}
#       columns {name: "MedInc" type: FLOAT}
#       columns {name: "MedHouseVal" type: FLOAT}
#     }
#   }
# }
#  to <class 'dict'>


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
