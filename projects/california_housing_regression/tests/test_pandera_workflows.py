import pytest
from hypothesis import given, settings

from flytekit import task

from california_housing_regression import (
    pandera_workflows,
    pandera_column_error_workflows,
    pandera_dtype_error_workflows,
    pandera_value_error_workflows,
    pandera_stats_error_workflows,
)


def test_dataset():
    kwargs = {"test_size": 0.2, "random_state": 100}
    pandera_workflows.get_dataset(**kwargs)

    for get_dataset_fn, error_regex in [
        (
            pandera_column_error_workflows.get_dataset,
            r"column 'Latitude' not in dataframe",
        ),
        (
            pandera_dtype_error_workflows.get_dataset,
            r"Could not coerce <class 'pandas.core.series.Series'> data_container into type float64",
        ),
        (
            pandera_value_error_workflows.get_dataset,
            r"failed element-wise validator 0:\s<Check in_range: in_range\(-90, 90\)>",
        ),
        (
            pandera_stats_error_workflows.get_dataset,
            r"MedHouseVal mean value is not equal to 2.0685 \[alpha=1e-3\]",
        )
    ]:
        with pytest.raises(TypeError, match=error_regex):
            get_dataset_fn(**kwargs)


@task
def summarize_dataset_error(dataset: pandera_workflows.Dataset) -> pandera_workflows.DatasetSummary:
    return dataset.describe().drop("Latitude", axis="columns")


@settings(max_examples=10)
@given(pandera_workflows.CaliforniaHousingData.strategy(size=30))
def test_summarize_data(dataset):
    pandera_workflows.summarize_dataset(dataset=dataset)

    with pytest.raises(TypeError, match=r"column 'Latitude' not in dataframe"):
        summarize_dataset_error(dataset=dataset)


@settings(max_examples=10)
@given(pandera_workflows.CaliforniaHousingData.strategy(size=30))
def test_train_model(dataset):
    model = pandera_workflows.train_model(
        dataset=dataset,
        hyperparameters=pandera_workflows.Hyperparameters(alpha=0.1, random_state=100)
    )

    features = dataset.drop(pandera_workflows.TARGET, axis="columns")
    predictions = model.predict(features)
    assert all(isinstance(x, float) for x in predictions)
