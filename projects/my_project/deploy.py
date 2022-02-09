import subprocess
from pathlib import Path

import docker
import git
import typer


app = typer.Typer()

docker_client = docker.from_env()


IMAGE_NAME = "flytelab"
REGISTRY = "ghcr.io/flyteorg"
PROJECT_NAME = "flytelab-my_project".replace("_", "-")
DESCRIPTION = "my flyte project"


def create_project(remote: bool):
    config = Path(".flyte") / f"{'remote' if remote else 'sandbox'}-config.yaml"
    output = subprocess.run(
        [
            "flytectl",
            "get",
            "project",
            PROJECT_NAME,
            "--config", config,
        ],
        capture_output=True,
    )
    if output.stdout.decode().strip():
        return

    typer.echo(f"Creating project {PROJECT_NAME}")
    subprocess.run(
        [
            "flytectl",
            "create",
            "project",
            "--project", PROJECT_NAME,
            "--name", PROJECT_NAME,
            "--id", PROJECT_NAME,
            "--description", DESCRIPTION,
            "--config", config,
        ]
    )


def get_version():
    repo = git.Repo(".", search_parent_directories=True)
    if repo.is_dirty():
        typer.echo("Please commit git changes before building", err=True)
        raise typer.Exit(code=1)
    commit = repo.rev_parse("HEAD")
    return commit.hexsha


def get_tag(version):
    return f"{REGISTRY}/{IMAGE_NAME}:{PROJECT_NAME}-{version}"


def sandbox_docker_build(tag):
    typer.echo("Building image in Flyte sandbox")
    subprocess.run([
        "flytectl",
        "sandbox",
        "exec",
        "--",
        "docker",
        "build",
        ".",
        "--tag",
        tag,
    ])


def docker_build(tag: str, remote: bool) -> docker.models.images.Image:
    client = docker.from_env()

    # TODO: image build, push, flytectl serialization and registration
    config = Path(".flyte") / f"{'remote' if remote else 'sandbox'}.config"

    typer.echo(f"Building image: {tag}")
    image, build_logs = client.images.build(
        path=".",
        dockerfile="Dockerfile",
        tag=tag,
        buildargs={
            "image": tag,
            "config": str(config),
        }
    )
    for line in build_logs:
        typer.echo(line)
    return image


def docker_push(image: docker.models.images.Image):
    for line in docker_client.api.push(image.tags[0], stream=True, decode=True):
        typer.echo(line)


def serialize(tag: str):
    typer.echo("Serializing Flyte workflows")
    subprocess.run([
        "pyflyte",
        "-c", "flyte.config",
        "--pkgs", "my_project",
        "package",
        "--force",
        "--in-container-source-path", "/root",
        "--image", tag
    ])


def register(version: str, remote: bool, domain: str):
    typer.echo("Registering Flyte workflows")
    config = Path(".flyte") / f"{'remote' if remote else 'sandbox'}-config.yaml"
    subprocess.run([
        "flytectl",
        "-c", config,
        "register",
        "files",
        "--project", PROJECT_NAME,
        "--domain", domain,
        "--archive", "flyte-package.tgz",
        "--force",
        "--version", version
    ])


@app.command()
def main(remote: bool = False, domain: str = "development"):
    create_project(remote)
    version = get_version()
    tag = get_tag(version)
    if remote:
        docker_push(docker_build(tag, remote))
    else:
        sandbox_docker_build(tag)
    serialize(tag)
    register(version, remote, domain)


if __name__ == "__main__":
    app()
