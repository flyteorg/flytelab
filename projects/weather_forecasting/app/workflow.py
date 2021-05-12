import os

from flytekit import task, workflow


@task
def greet(name: str) -> str:
    return f"Hello there, {name}"


@workflow
def hello_world(name: str = "world") -> str:
    greeting = greet(name=name)
    return greeting


@workflow
def get_api_key() -> str:
    api_key = os.getenv("NOAA_API_KEY")
    return api_key
