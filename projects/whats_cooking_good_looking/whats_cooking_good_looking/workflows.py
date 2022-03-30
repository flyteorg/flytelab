import json
import random
from pathlib import Path
from typing import List

import pandas as pd
import spacy
from flytekit import Resources, dynamic, task, workflow
from snscrape.modules.twitter import TwitterSearchScraper
from spacy.language import Language
from spacy.training import Example
from spacy.util import compounding, minibatch

SPACY_MODEL = {"en": "en_core_web_sm"}

CACHE_VERSION = "2.2"
request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")


def load_config():
    """ Load config """
    config_file_path = (Path(__file__).parent.resolve() / 'config.json')
    with open(config_file_path, "r") as f:
        config = json.load(f)
        print(f"Loaded config: {config}")
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


def retrieve_train_data(train_data_path: str) -> List:
    """ Load jsonl train data as a list, ready to be ingested by spacy model.

    Args:
        train_data_local_path (str): Path of files to load.

    Returns:
        List: Tuple of texts and dict of entities to be used for training.
    """
    train_data_df = pd.read_json(train_data_path, lines=True)
    train_data = []
    for _, row in train_data_df.iterrows():
        train_text = row["text"]
        train_entities = {"entities": [tuple(entity_elt) for entity_elt in row["entities"]]}
        formatted_train_line = (train_text, train_entities)
        train_data.append(formatted_train_line)
    if train_data:
        print("Training data has been retrieved from GCS !")
    else:
        print("No training data could be retrieved from GCS...")
    return train_data


@task
def train_model(train_data_gs_uri: str, nlp: Language, training_iterations: int = 30) -> Language:
    """ Uses new labelled data to improve spacy NER model.

    Args:
        train_data_gs_uri (str): GCS Uri of data file to be used for training. After being loaded, format \
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
    print("Preparing model training")
    train_data = retrieve_train_data(train_data_gs_uri)
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
    return nlp


@dynamic(
    cache=False,
    requests=request_resources,
    limits=limit_resources,
)
def init_model(
    train_data_gs_uri: str,
    training_iterations: int = 30,
    lang: str = "en"
) -> Language:
    """ Initialize Spacy Model. If train data is available on GCS bucket, train a model.

    Args:
        train_data_gs_uri (str): GCS Uri of data file to be used for training.
        training_iterations (int, optional): Number of training iterations. Defaults to 30.
        lang (str, optional): Language of Spacy model and texts. Defaults to "en".

    Returns:
        Language: Spacy model, trained if train data has been downloaded.
    """
    nlp = spacy.load(SPACY_MODEL[lang])
    if train_data_gs_uri:
        nlp = train_model(train_data_gs_uri=train_data_gs_uri, nlp=nlp, training_iterations=training_iterations)
    return nlp


@task
def apply_model(nlp: Language, tweets_list: str) -> str:
    """Applies spacy model to each tweet to extract entities from.

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
    tweets_list = get_tweets_list(
        keyword_list=config["keyword_list"], lang=config["lang"], max_results=config["max_results"]
    )
    nlp = init_model(
        train_data_gs_uri=config["train_data_gs_uri"],
        training_iterations=config["training_iterations"],
        lang=config["lang"]
    )
    return apply_model(nlp=nlp, tweets_list=tweets_list)


if __name__ == "__main__":
    print(f"trained model: {main()}")
