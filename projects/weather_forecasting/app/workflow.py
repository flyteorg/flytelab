from flytekit import task, workflow


@task
def greet(name: str) -> str:
    return f"Hello, {name}"


@workflow
def hello_world(name: str = "world") -> str:
    greeting = greet(name=name)
    return greeting
