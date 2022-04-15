import json
from pathlib import Path
from typing import List

import spacy
from flytekit import Resources, task, workflow
from snscrape.modules.twitter import TwitterSearchScraper
from whats_cooking_good_looking.utils import (doc_to_spans, download_from_gcs,
                                              load_config, upload_to_gcs)

SPACY_MODEL = {"en": "en_core_web_sm"}

CACHE_VERSION = "2.2"
request_resources = Resources(cpu="1", mem="500Mi", storage="500Mi")
limit_resources = Resources(cpu="2", mem="1000Mi", storage="1000Mi")


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


@task
def load_model(
    lang: str,
    from_gcs: bool,
    gcs_bucket: str,
    gcs_source_blob_name: str,
) -> spacy.Language:
    """Loads spacy model either from gcs if specified or given the source language.

    Args:
        lang (str): Language in which tweets must be written(iso-code).
        from_gcs (bool): True if needs to download custom spacy model from gcs.
        gcs_bucket (str): bucket name where to retrieve spacy model if from_gcs.
        gcs_source_blob_name (str): blob name where to retrieve spacy model if from_gcs.

    Returns:
        Language: spacy model.
    """
    if from_gcs:
        Path("tmp").mkdir(parents=True, exist_ok=True)
        output_filename = download_from_gcs(
            gcs_bucket, gcs_source_blob_name, "tmp", explicit_filepath=True
        )[0]
        nlp = spacy.load(output_filename)
    else:
        model_name = SPACY_MODEL[lang]
        nlp = spacy.load(model_name)
    return nlp


@task
def apply_model(
    nlp: bytes, tweets_list: str, bucket_name: str, source_blob_name: str
) -> str:
    """Applies spacy model to each tweet to extract entities from and convert them into
    Label studio task format.

    Args:
        nlp (Language): Spacy model to use for inference.
        tweets_list (str): json dumped list of tweets.
        bucket_name (str): Name of the GCS bucket to upload to.
        source_blob_name (str): File name of GCS uploaded file.

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
    labelstudio_tasks = []
    model_name = SPACY_MODEL["en"]
    for tweet in json.loads(tweets_list):
        predictions = []
        text = tweet["text"]
        doc = nlp(text)
        spans, ents = doc_to_spans(doc)
        entities |= ents
        predictions.append({"model_version": model_name, "result": spans})
        labelstudio_tasks.append({"data": {"text": text}, "predictions": predictions})
    with open("tasks.json", mode="w") as f:
        json.dump(labelstudio_tasks, f, indent=2)
    json_labelstudio_tasks = json.dumps(labelstudio_tasks)
    upload_to_gcs(
        bucket_name, source_blob_name, json_labelstudio_tasks, content_type=None
    )
    return json_labelstudio_tasks


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
    config = load_config("apply")
    tweets_list = get_tweets_list(
        keyword_list=config["keyword_list"],
        lang=config["lang"],
        max_results=config["max_results"],
    )
    nlp = load_model(
        lang=config["lang"],
        from_gcs=config["from_gcs"],
        gcs_bucket=config["bucket_name"],
        gcs_source_blob_name=config["gcs_spacy_model_blob_name"],
    )
    return apply_model(
        nlp=nlp,
        tweets_list=tweets_list,
        bucket_name=config["bucket_name"],
        source_blob_name=config["applied_model_output_blob_name"],
    )


if __name__ == "__main__":
    print(f"Applied model: {main()}")
