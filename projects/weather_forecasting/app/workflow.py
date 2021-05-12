import os

from flytekit import task, workflow


@task
def greet(name: str) -> str:
    return f"Hello there, {name}"


@workflow
def hello_world(name: str = "world") -> str:
    greeting = greet(name=name)
    return greeting


@task
def fetch_key(key: str) -> str:
    return os.getenv(key)


@workflow
def get_api_key(key: str) -> str:
    return fetch_key(key=key)
