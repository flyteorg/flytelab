import json
import os
from itertools import groupby
from pathlib import Path
from typing import List

from google.cloud import storage


def load_config(train_or_apply: str) -> dict:
    """Load config"""
    config_file_path = Path(__file__).parent.resolve() / "config.json"
    with open(config_file_path, "r") as f:
        config = json.load(f)
        print(f"Loaded config: {config}")
    return config[train_or_apply]


def doc_to_spans(doc):
    """This function converts spaCy docs to the list of named entity spans in Label Studio compatible JSON format"""
    tokens = [(tok.text, tok.idx, tok.ent_type_) for tok in doc]
    results = []
    entities = set()
    for entity, group in groupby(tokens, key=lambda t: t[-1]):
        if not entity:
            continue
        group = list(group)
        _, start, _ = group[0]
        word, last, _ = group[-1]
        text = " ".join(item[0] for item in group)
        end = last + len(word)
        results.append(
            {
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {"start": start, "end": end, "text": text, "labels": [entity]},
            }
        )
        entities.add(entity)

    return results, entities


def load_train_data(train_data_files: str) -> List:
    """Load jsonl train data as a list, ready to be ingested by spacy model.

    Args:
        train_data_local_path (str): Path of files to load.

    Returns:
        List: Tuple of texts and dict of entities to be used for training.
    """
    train_data = []
    for data_file in train_data_files:
        with open(data_file, "r") as f:
            for json_str in list(f):
                train_data_dict = json.loads(json_str)
                train_text = train_data_dict["text"]
                train_entities = {
                    "entities": [
                        tuple(entity_elt) for entity_elt in train_data_dict["entities"]
                    ]
                }
                formatted_train_line = (train_text, train_entities)
                train_data.append(formatted_train_line)
    return train_data


def download_from_gcs(
    bucket_name: str,
    source_blob_name: str,
    destination_folder: str,
    explicit_filepath: bool = False,
) -> str:
    """Download gcs data locally.

    Args:
        bucket_name (str): Name of the GCS bucket.
        source_blob_name (str): GCS path to data in the bucket.
        destination_folder (str): Folder to download GCS data to.

    Returns:
        str: Local destination folder
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)
    filepath_list = []
    for blob in blobs:
        if not blob.name.endswith("/"):
            filename = blob.name.replace("/", "_")
            local_path = os.path.join(destination_folder, filename)
            blob.download_to_filename(local_path)
            filepath_list.append(local_path)
    print(f"Downloaded at {destination_folder}")
    if explicit_filepath:
        return filepath_list
    return destination_folder


def upload_to_gcs(bucket_name, source_blob_name, data, content_type=None):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.upload_from_string(data, content_type=content_type)
