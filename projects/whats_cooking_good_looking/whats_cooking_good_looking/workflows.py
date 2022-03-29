import glob
import json
import os
import random
from typing import List

import spacy
from flytekit import Resources, dynamic, task, workflow
from google.cloud import storage
from snscrape.modules.twitter import TwitterSearchScraper
from spacy.language import Language
from spacy.training import Example
from spacy.util import compounding, minibatch
from itertools import groupby


SPACY_MODEL = {"en": "en_core_web_sm"}

CACHE_VERSION = "2.2"
request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")


def load_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config


@task
def get_tweets_list(
    keyword_list: List[str], lang: str = "en", max_results: int = 1000
) -> str:
    """Collects `max_results` tweets mentioning any of the words in `keywords_list` written in language `lang`.

    Args:
        keyword_list (List[str]): List of keywords that tweets must mention at least one of.
        lang (str, optional): Language in which tweets must be written(iso-code). Defaults to "en".
        max_results (int, optional): Number of maximum tweets to retrieve. Defaults to 1000.

    Returns:
        str: json dumped results with following shape
            [
                {
                    "date": "2022-03-25 16:23:01+00:00,
                    "tweet_id": "XXXXXXX",
                    "text": "some tweet",
                    "username": "some user"
                },
            ]
    """
    keywords_query = " OR ".join(keyword_list)
    query = f"({keywords_query}) lang:{lang}"
    tweets_list = []
    for tweet_idx, tweet_post in enumerate(TwitterSearchScraper(query).get_items()):
        if tweet_idx == max_results:
            break
        tweets_list.append(
            {
                "date": str(tweet_post.date),
                "tweet_id": str(tweet_post.id),
                "text": str(tweet_post.content),
                "username": str(tweet_post.username),
            }
        )
    return json.dumps(tweets_list)


def download_from_gcs(bucket_name: str, source_blob_name: str) -> str:
    """ Download gcs data locally.

    Args:
        bucket_name (str): Name of the GCS bucket.
        source_blob_name (str): GCS path to data in the bucket.

    Returns:
        str: Local destination folder
    """
    destination_folder = os.path.join(os.getcwd(), "train_data")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name, delimiter="/")
    for blob in blobs:
        if not blob.name.endswith("/"):
            filename = blob.name.replace("/", "_")
            blob.download_to_filename(os.path.join(destination_folder, filename))
    print(f"Downloaded at {destination_folder}")
    return destination_folder


def load_train_data(train_data_files: str) -> List:
    """ Load jsonl train data as a list, ready to be ingested by spacy model.

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
                train_entities = {"entities": [tuple(entity_elt) for entity_elt in train_data_dict["entities"]]}
                formatted_train_line = (train_text, train_entities)
                train_data.append(formatted_train_line)
    return train_data


@task
def retrieve_train_data_path(bucket_name: str, train_data_gcs_folder: str) -> List[str]:
    """ Retrieves training data from GCS.

    Args:
        bucket_name (str): Name of the GCS bucket.
        train_data_gcs_folder (str): GCS folder containing train data.

    Returns:
        List: Tuple of texts and dict of entities to be used for training.
    """
    train_data_local_path = download_from_gcs(bucket_name, train_data_gcs_folder)
    train_data_files = glob.glob(os.path.join(train_data_local_path, "*.jsonl"))
    return train_data_files


@task
def train_model(train_data_files: List[str], nlp: Language, n_iterations: int = 30) -> Language:
    """ Uses new labelled data to improve spacy NER model.

    Args:
        train_data_files (List[str]): List of data filepath to train model on. After being loaded, format \
            should be the following:
                train_data = [
                    ("Text to detect Entities in.", {"entities": [(15, 23, "PRODUCT")]}),
                    ("Flyte is another example of organisation.", {"entities": [(0, 6, "ORG")]}),
                ]
        nlp (Language): Spacy base model to train on.
        n_iterations (int): Number of training iterations to make. Defaults to 30.
        lang (str, optional): Texts language. Defaults to "en".

    Returns:
        Language: Trained spacy model
    """
    train_data = load_train_data(train_data_files)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = spacy.blank("en").initialize()
        for iteration in range(n_iterations):
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
    n_iterations: int = 30,
    lang: str = "en",
) -> Language:
    """ Initialize Spacy Model. If train data is available on GCS bucket, train a model.

    Args:
        bucket_name (str): Name of the bucket to retrieve training data from.
        train_data_gcs_folder (str): Path to training data folder in GCS bucket.
        n_iterations (int, optional): Number of training iterations. Defaults to 30.
        lang (str, optional): Language of Spacy model and texts. Defaults to "en".

    Returns:
        Language: Spacy model, trained if train data has been downloaded.
    """
    nlp = spacy.load(SPACY_MODEL[lang])
    train_data_files = retrieve_train_data_path(bucket_name=bucket_name, train_data_gcs_folder=train_data_gcs_folder)
    if train_data_files:
        print("Performing model training with downloaded training data...")
        nlp = train_model(train_data_files=train_data_files, nlp=nlp, n_iterations=n_iterations)
        print("Spacy model has been trained !")
    return nlp


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
        text = ' '.join(item[0] for item in group)
        end = last + len(word)
        results.append({
            'from_name': 'label',
            'to_name': 'text',
            'type': 'labels',
            'value': {
                'start': start,
                'end': end,
                'text': text,
                'labels': [entity]
            }
        })
        entities.add(entity)

    return results, entities


@task
def apply_model(nlp: Language, tweets_list: str) -> str:
    """Applies spacy model to each tweet to extract entities from and convert them into 
    Label studio task format.

    Args:
        nlp (Language): Spacy model to use for inference.
        tweets_list (str): json dumped list of tweets.

    Returns:
        str: json dumped results with following shape
            [
                {
                    "date": "2022-03-25 16:23:01+00:00,
                    "tweet_id": "XXXXXXX",
                    "text": "some tweet",
                    "username": "some user"
                    "entities": [
                        {
                        "label": "some label",
                        "start_char": "index beginning char entity",
                        "end_char": "index end char entity"
                        },
                    ]

                }
            ]
    """
    entities = set()
    tasks = []
    model_name = SPACY_MODEL['en']
    for tweet in json.loads(tweets_list):
        predictions = []
        text = tweet["text"]
        doc = nlp(text)
        spans, ents = doc_to_spans(doc)
        entities |= ents
        predictions.append({'model_version': model_name, 'result': spans})
        tasks.append({
            'data': {'text': text},
            'predictions': predictions
        })
    with open('tasks.json', mode='w') as f:
        json.dump(tasks, f, indent=2)
    return json.dumps(tasks)


@workflow
def main() -> str:
    """Main workflow searching for entities in beauty related tweets.

    Returns:
        str: json dumped results with following shape
            [
                {
                    "date": "2022-03-25 16:23:01+00:00,
                    "tweet_id": "XXXXXXX",
                    "text": "some tweet",
                    "username": "some user"
                    "entities": [
                        {
                        "label": "some label",
                        "start_char": "index beginning char entity",
                        "end_char": "index end char entity"
                        },
                    ]

                }
            ]
    """
    config = load_config()
    tweets_list = get_tweets_list(
        keyword_list=config["keyword_list"], lang=config["lang"], max_results=config["max_results"]
    )
    nlp = init_model(
        bucket_name=config["bucket_name"],
        train_data_gcs_folder=config["train_data_gcs_folder"],
        n_iterations=10,
        lang=config["lang"]
    )
    return apply_model(nlp=nlp, tweets_list=tweets_list)




if __name__ == "__main__":
    print(f"trained model: {main()}")
