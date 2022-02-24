import os
import subprocess
import uuid
from pathlib import Path

import docker
import git
import typer


app = typer.Typer()

docker_client = docker.from_env()


IMAGE_NAME = "flytelab"
REGISTRY = "ghcr.io/cosmicBboy".lower()
PROJECT_NAME = "flytelab-pytorch_gpu".replace("_", "-").lower()
DESCRIPTION = "A flytelab project"


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
        check=True,
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
        ],
        check=True,
    )


def get_version(fast: bool):
    repo = git.Repo(".", search_parent_directories=True)
    if not fast and repo.is_dirty():
        typer.echo(
            "Please commit git changes before building. If you haven't updated any system/python dependencies "
            "but want to deploy task/workflow code changes, use the --fast flag to do fast registration.",
            err=True
        )
        raise typer.Exit(code=1)
    commit = repo.rev_parse("HEAD")
    return commit.hexsha


def get_tag(version, registry=None):
    return f"{REGISTRY if registry is None else registry}/{IMAGE_NAME}:{PROJECT_NAME}-{version}"


def sandbox_docker_build(tag):
    typer.echo("Building image in Flyte sandbox")
    subprocess.run(
        [
            "flytectl",
            "sandbox",
            "exec",
            "--",
            "docker",
            "build",
            ".",
            "--tag",
            tag,
        ],
        check=True,
    )


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


def serialize(tag: str, remote: bool, fast: bool):
    typer.echo("Serializing Flyte workflows")
    config = Path(".flyte") / f"{'remote' if remote else 'sandbox'}.config"
    package = Path(".") / "flyte-package.tgz"
    if package.exists():
        os.remove(package)
    subprocess.run(
        [
            "pyflyte",
            "-c", str(config),
            "--pkgs", "pytorch_gpu",
            "package",
            "--force",
            "--image", tag,
            *(
                ["--fast"]
                if fast
                else ["--in-container-source-path", "/root"]
            ),
        ],
        check=True,
        # inject the FLYTE_SANDBOX environment variable to the serialization runtime
        env={"FLYTE_SANDBOX": "1" if not remote else "0", **os.environ},
    )


def register(version: str, remote: bool, fast: bool, domain: str):
    typer.echo("Registering Flyte workflows")
    config = Path(".flyte") / f"{'remote' if remote else 'sandbox'}-config.yaml"
    if fast:
        version = f"{version}-fast{uuid.uuid4().hex[:7]}"
    subprocess.run(
        [
            "flytectl",
            "-c", config,
            "register",
            "files",
            "--project", PROJECT_NAME,
            "--domain", domain,
            "--archive", "flyte-package.tgz",
            "--force",
            "--version", version
        ],
        check=True,
    )
    typer.echo(f"Successfully registered version {version}")


@app.command()
def main(remote: bool = False, fast: bool = False, domain: str = "development", registry: str = None):
    if remote and fast:
        typer.echo(
            "Fast registration is not enabled when deploying to remote. "
            "Please deploy your workflows without the --fast flag.",
            err=True
        )
    create_project(remote)
    version = get_version(fast)
    tag = get_tag(version, registry)
    if not fast:
        if remote:
            docker_push(docker_build(tag, remote))
        else:
            sandbox_docker_build(tag)
    serialize(tag, remote, fast)
    register(version, remote, fast, domain)


if __name__ == "__main__":
    app()
