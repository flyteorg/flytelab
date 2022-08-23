import json
import pickle
import random
from collections import defaultdict

import spacy
from flytekit import Resources, dynamic, task, workflow
from spacy.language import Language
from spacy.training import Example
from spacy.util import compounding, minibatch

from whats_cooking_good_looking.apply_ner_workflow import load_model
from whats_cooking_good_looking.utils import (download_bytes_from_gcs,
                                              load_config, upload_to_gcs)

SPACY_MODEL = {"en": "en_core_web_sm"}

CACHE_VERSION = "2.2"
request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")

THRESHOLD_ACCURACY = 0.7


@task
def evaluate_ner(labelstudio_tasks: bytes) -> dict:
    """Computes accuracy, precision and recall of NER model out of label studio output.

    Args:
        labelstudio_tasks (list): List of dicts outputs of label studio annotation with following format
                        [
                            {
                            "result": [
                                {
                                    "value": {"start": 10, "end": 17, "text": "Chennai", "labels": ["LOC"]},
                                    "from_name": "label",
                                    "to_name": "text",
                                    "type": "labels",
                                    "origin": "manual",
                                }
                            ],
                            "predictions": [
                                    {
                                    "result": {"start": 10, "end": 17, "text": "Chennai", "labels": ["LOC"]},
                                    "model_version": "en_core_web_sm",
                                    }
                                ],
                            }
                        ]

    Returns:
        dict: mapping {model_name: accuracy}

    """
    model_acc = dict()
    model_hits = defaultdict(int)
    for ls_task in json.loads(labelstudio_tasks):
        annotation_result = ls_task["result"][0]["value"]
        for key in annotation_result:
            annotation_result.pop("id", None)
        for prediction in ls_task["predictions"]:
            model_version = prediction["model_version"]
            model_hits[model_version] += int(prediction["result"] == annotation_result)

    num_task = len(labelstudio_tasks)
    for model_name, num_hits in model_hits.items():
        acc = num_hits / num_task
        model_acc[model_name] = acc
        print(f"Accuracy for {model_name}: {acc:.2f}%")
    return model_acc


@task
def load_tasks(bucket_name: str, source_blob_name: str) -> bytes:
    """Loads Label Studio annotations.

    Args:
        bucket_name (str): GCS bucket name where tasks are stored.
        source_blob_name (str): GCS blob name where tasks are stored.

    Returns:
        str: json dumped tasks
    """
    labelstudio_tasks = download_bytes_from_gcs(
        bucket_name=bucket_name, source_blob_name=source_blob_name
    )
    return labelstudio_tasks


@task
def format_tasks_for_train(labelstudio_tasks: bytes) -> str:
    """Format Label Studio output to be trained in spacy custom model.

    Args:
        labelstudio_tasks (str): json dumped labelstudio_tasks

    Returns:
        str: json dumped train data formatted
    """
    train_data = []
    for ls_task in json.loads(labelstudio_tasks):
        entities = [
            (ent["value"]["start"], ent["value"]["end"], label)
            for ent in ls_task["result"]
            for label in ent["value"]["labels"]
        ]
        if entities != []:
            train_data.append((ls_task["task"]["data"]["text"], {"entities": entities}))
    return json.dumps(train_data)


@task
def train_model(
    train_data: str,
    nlp: Language,
    training_iterations: int,
    bucket_out: str,
    source_blob_name: str,
) -> Language:
    """ Uses new labelled data to improve spacy NER model. Uploads trained model in GCS.

    Args:
        train_data_files (List[str]): List of data filepath to train model on. After being loaded, format \
            should be the following:
                train_data = [
                    ("Text to detect Entities in.", {"entities": [(15, 23, "PRODUCT")]}),
                    ("Flyte is another example of organisation.", {"entities": [(0, 6, "ORG")]}),
                ]
        nlp (Language): Spacy base model to train on.
        training_iterations (int): Number of training iterations to make. Defaults to 30.

    Returns:
        Language: Trained spacy model
    """
    train_data = json.loads(train_data)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    print("Starting model training")
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = spacy.blank("en").initialize()
        for iteration in range(training_iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.35, losses=losses, sgd=optimizer)
                    print("Iteration n°", iteration)
                    print("Losses", losses)
    print("Model training completed !")
    upload_to_gcs(bucket_out, source_blob_name, pickle.dumps(nlp))
    print("Model upload on GCS completed !")
    return nlp


@dynamic(
    cache=False,
    requests=request_resources,
    limits=limit_resources,
)
def train_model_if_necessary(
    labelstudio_tasks: bytes,
    metrics_dict: dict,
    model_name: str,
    training_iterations: int,
    bucket_out: str,
    model_output_blob_name: str,
):
    """Checks for model accuracy. If it's high enough, the pipeline stops, else it trains a new model \
        and upload it to GCS.

    Args:
        labelstudio_tasks (bytes): Label studio annotations
        metrics_dict (dict): mapping between model name and accuracy
        model_name (str): model name from which we get accuracy
        training_iterations (int): number of training iterations for the spacy NER model
    """
    if metrics_dict[model_name] >= THRESHOLD_ACCURACY:
        print(f"No need to train. Accuracy of {metrics_dict[model_name]} is above threshold {THRESHOLD_ACCURACY}")
    else:
        train_data = format_tasks_for_train(labelstudio_tasks=labelstudio_tasks)
        nlp = load_model(
            lang="en",
            from_gcs=False,
            gcs_bucket=bucket_out,
            gcs_source_blob_name=model_output_blob_name,
        )
        nlp = train_model(
            train_data=train_data,
            nlp=nlp,
            training_iterations=training_iterations,
            bucket_out=bucket_out,
            source_blob_name=model_output_blob_name,
        )


@workflow
def main():
    """Main training workflow evaluating model based on labelled observations.
    * If accuracy is high enough, the pipeline ends
    * If accuracy is below threshold, the pipeline trains a new model based on those
      observations and dumps it on GCS
    """
    config = load_config("train")
    labelstudio_tasks = load_tasks(
        bucket_name=config["bucket_label_out_name"],
        source_blob_name=config["label_studio_output_blob_name"],
    )
    metrics_dict = evaluate_ner(labelstudio_tasks=labelstudio_tasks)
    train_model_if_necessary(
        labelstudio_tasks=labelstudio_tasks,
        metrics_dict=metrics_dict,
        training_iterations=config["training_iterations"],
        model_name=config["model_name"],
        bucket_out=config["bucket_name"],
        model_output_blob_name=config["model_output_blob_name"],
    )


if __name__ == "__main__":
    main()
