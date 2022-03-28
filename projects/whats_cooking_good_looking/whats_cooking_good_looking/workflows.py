import json
from typing import List

import spacy
from flytekit import task, workflow
from snscrape.modules.twitter import TwitterSearchScraper

SPACY_MODEL = {"en": "en_core_web_trf"}


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


@task
def apply_model(tweets_list: str, lang: str = "en") -> str:
    """Applies spacy model to each tweet to extract entities from.

    Args:
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
    nlp = spacy.load(SPACY_MODEL[lang])
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
    return apply_model(tweets_list=tweets_list, lang=config["lang"])


if __name__ == "__main__":
    print(f"trained model: {main()}")
