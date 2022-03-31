import glob
import json
import os
import random
from pathlib import Path
from typing import List
from collections import defaultdict

import spacy
from flytekit import Resources, dynamic, task, workflow
from spacy.language import Language
from spacy.training import Example
from spacy.util import compounding, minibatch

from utils import download_from_gcs, doc_to_spans, load_config, load_train_data

SPACY_MODEL = {"en": "en_core_web_sm"}

CACHE_VERSION = "2.2"
request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")


@task
def retrieve_train_data_path(bucket_name: str, train_data_gcs_folder: str) -> List[str]:
    """Retrieves training data from GCS.

    Args:
        bucket_name (str): Name of the GCS bucket.
        train_data_gcs_folder (str): GCS folder containing train data.

    Returns:
        List: Tuple of texts and dict of entities to be used for training.
    """
    train_data_local_folder = Path(__file__).parent.parent.resolve() / "train_data"
    download_from_gcs(bucket_name, train_data_gcs_folder, train_data_local_folder)
    train_data_files = glob.glob(os.path.join(train_data_local_folder, "*.jsonl"))
    return train_data_files

@task
def evaluate_ner(tasks: List[dict]) -> dict:
    """Computes accuracy, precision and recall of NER model out of label studio output.

    Args:
        tasks (list): List of dicts outputs of label studio annotation with following format
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
                                    "model_version": "dummy",
                                    }
                                ],
                            }
                        ]

    Returns:
        dict: mapping {model_name: accuracy}
    
    """
    model_acc = dict()
    model_hits = defaultdict(int)
    for task in tasks:
        annotation_result = task["result"][0]["value"]
        for key in annotation_result:
            if key == "id":
                annotation_result.pop("id")
        for prediction in task["predictions"]:
            model_version = prediction["model_version"]
            model_hits[model_version] += int(prediction["result"] == annotation_result)

    num_task = len(tasks)
    for model_name, num_hits in model_hits.items():
        acc = num_hits / num_task
        model_acc[model_name] = acc
        print(f"Accuracy for {model_name}: {acc:.2f}%")
    return model_acc


@task
def load_tasks(bucket_name: str, source_blob_name: str) -> str:
    Path("tmp").mkdir(parents=True, exist_ok=True)
    local_folder = download_from_gcs(bucket_name=bucket_name, source_blob_name=source_blob_name, destination_folder="tmp")
    tasks = []
    for filename in os.listdir(local_folder):
        with open(os.path.join(local_folder, filename), "r") as f:
            annotations = json.load(f)
        if isinstance(annotations, dict):
            tasks.append(annotations)
        elif isinstance(annotations, list):
            tasks.extend(annotations)
    return json.dumps(tasks)


@task
def format_tasks_for_train(tasks: str):
    train_data = []
    for task in json.loads(tasks):
        entities = [(ent["start"], ent["end"], label) for ent in task["results"] for label in ent["labels"]]
        if entities != []:
            train_data.append((task["train"]["data"]["text"], {"entities": entities}))
    return json.dumps(train_data)


@task
def train_model(
    train_data: str, nlp: Language, training_iterations: int = 30
) -> Language:
    """ Uses new labelled data to improve spacy NER model.

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
    return nlp


@dynamic(
    cache=False,
    requests=request_resources,
    limits=limit_resources,
)
def init_model(
    bucket_name: str,
    train_data_gcs_folder: str,
    training_iterations: int = 30,
    lang: str = "en",
) -> Language:
    """Initialize Spacy Model. If train data is available on GCS bucket, train a model.

    Args:
        bucket_name (str): Name of the bucket to retrieve training data from.
        train_data_gcs_folder (str): Path to training data folder in GCS bucket.
        training_iterations (int, optional): Number of training iterations. Defaults to 30.
        lang (str, optional): Language of Spacy model and texts. Defaults to "en".

    Returns:
        Language: Spacy model, trained if train data has been downloaded.
    """
    nlp = spacy.load(SPACY_MODEL[lang])
    train_data_files = retrieve_train_data_path(
        bucket_name=bucket_name, train_data_gcs_folder=train_data_gcs_folder
    )
    if train_data_files:
        print("Performing model training with downloaded training data...")
        nlp = train_model(
            train_data_files=train_data_files,
            nlp=nlp,
            training_iterations=training_iterations,
        )
        print("Spacy model has been trained !")
    return nlp


@workflow
def main():
    config = load_config("train")
    tasks = load_tasks(config["gcs_bucket_in"], config["gcs_blob_in"])
    metrics_dict = evaluate_ner(tasks)
    train_data = format_tasks_for_train(tasks)
    train_model(train_data)
    return 


if __name__ == "__main__":
    print(f"Trained model: {main()}")
