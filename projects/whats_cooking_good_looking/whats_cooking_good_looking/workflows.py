import json
from typing import List, Tuple

import glob
import os
from google.cloud import storage
from sklearn.pipeline import Pipeline
import spacy
from flytekit import task, workflow, Resources, dynamic
from snscrape.modules.twitter import TwitterSearchScraper

import random
from spacy.language import Language
from spacy.util import minibatch, compounding
from pathlib import Path


SPACY_MODEL = {"en": "en_core_web_trf"}

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
                "username": str(tweet_post.user.username),
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
    destination_file_name = os.path.join(os.getcwd(), "train_data")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded at {destination_file_name}")
    return destination_file_name


def load_train_data(train_data_local_path: str) -> List[Tuple[str, dict]]:
    """ Load txt train data as a list.

    Args:
        train_data_local_path (str): Path of files to load.

    Returns:
        List[Tuple[str, dict]]: Tuple of texts and dict of entities to be used for training.
    """
    train_data_files = glob.glob(os.path.join(train_data_local_path, "*.jsonl"))
    train_data = []
    for data_file in train_data_files:
        with open(data_file, "r") as f:
            for json_str in list(f):
                train_data_dict = json.loads(json_str)
                train_text = train_data_dict.pop["text"]
                formatted_train_line = (train_text, train_data_dict)
                train_data.append(formatted_train_line)
    return train_data


@task
def retrieve_train_data(bucket_name: str, train_data_gcs_folder: str) -> List[Tuple[str, dict]]:
    """ Retrieves training data from GCS.

    Args:
        bucket_name (str): Name of the GCS bucket.
        train_data_gcs_folder (str): GCS folder containing train data.

    Returns:
        List[Tuple[str, dict]]: Tuple of texts and dict of entities to be used for training.
    """
    train_data_local_path = download_from_gcs(bucket_name, train_data_gcs_folder)
    train_data = load_train_data(train_data_local_path)
    return train_data


@task
def train_model(train_data: List[Tuple[str, dict]], base_model: Language, n_iterations: int = 30, lang: str = "en") -> str:
    """ Uses new labelled data to improve spacy NER model.

    Args:
        train_data (List[dict]): List of data to train model on. Format should be the following: \
            train_data = [
                ("Text to detect Entities in.", {"entities": [(15, 23, "PRODUCT")]}),
                ("Flyte is another example of organisation.", {"entities": [(0, 6, "ORG")]}),
            ]
        base_model (Language): Spacy base model to train on.
        n_iterations (int): Number of training iterations to make. Defaults to 30.
        lang (str, optional): Texts language. Defaults to "en".
    """
    nlp = spacy.load(SPACY_MODEL[lang])
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*unaffected_pipes):
        for iteration in range(n_iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))  
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.5,
                    losses=losses,
                )
                print("Iteration n°", iteration)
                print("Losses", losses)
    output_dir = Path('/content/')
    nlp.to_disk(output_dir)
    return output_dir


@dynamic(
    cache=True,
    cache_version=CACHE_VERSION,
    requests=request_resources,
    limits=limit_resources,
)
def init_model(
    bucket_name: str,
    train_data_gcs_folder: str,
    n_iterations: int = 30,
    lang: str = "en",
):
    """ Initialize Spacy Model. If train data is available on GCS bucket, train a model.

    Args:
        bucket_name (str): Name of the bucket to retrieve training data from.
        train_data_gcs_folder (str): Path to training data folder in GCS bucket.
        n_iterations (int, optional): Number of training iterations. Defaults to 30.
        lang (str, optional): Language of Spacy model and texts. Defaults to "en".

    Returns:
        Language: Spacy model
    """
    nlp = spacy.load(SPACY_MODEL[lang])
    train_data = retrieve_train_data(bucket_name, train_data_gcs_folder)
    if train_data:
        trained_model_path = train_model(train_data, nlp, n_iterations, lang)
        nlp = spacy.load(trained_model_path)
    return nlp

@task
def apply_model(nlp, tweets_list: str, lang: str = "en") -> str:
    """Applies spacy model to each tweet to extract entities from.

    Args:
        nlp (Language): Spacy model to use for inference.
        tweets_list (str): json dumped list of tweets.
        lang (str, optional): Language in which tweets must be written(iso-code). Defaults to "en".

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
    tweet_entities = []
    for tweet in json.loads(tweets_list):
        tweet["entities"] = json.dumps(
            [
                {
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
                for ent in nlp(tweet["text"]).ents
            ]
        )
        tweet_entities.append(tweet)
    return json.dumps(tweet_entities)


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
    tweets_list = get_tweets_list(**config)
    nlp = init_model(
        config["bucket_name"], config["train_data_gcs_folder"], 30, SPACY_MODEL[config["lang"]]
    )
    return apply_model(nlp, tweets_list=tweets_list, lang=config["lang"])


if __name__ == "__main__":
    print(f"trained model: {main()}")
