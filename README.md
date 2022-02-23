<html>
    <p align="center"> 
        <img src="https://github.com/flyteorg/flyte/blob/master/rsts/images/flyte_circle_gradient_1_4x4.png" alt="Flyte Logo" width="100">
    </p>
    <h1 align="center">
        Flytelab
    </h1>
    <p align="center">
        The Open Source Repository of Flyte-based Projects
    </p>
    <p align="center">
        <a href="https://slack.flyte.org">
            <img src="https://img.shields.io/badge/slack-join_chat.svg?logo=slack&style=social" alt="Slack" />
        </a>
    </p>
</html>

The purpose of this repository is to showcase [Flyte's](https://flyte.org/) capabilities in end-to-end
applications that do some form of data processing or machine learning.

The source code for each project can be found in the `projects` directory, where each project has its
own set of dependencies.

## Table of Contents

- [Create a New Project](#-create-a-new-project)
- [Environment Setup](#-environment-setup)
- [Deployment](#-deployment)
- [Streamlit App [Optional]](#-streamlit-app-optional)

## 🚀 Create a New Project

Fork the repo on github, then clone it:

```bash
git clone https://github.com/<your-username>/flytelab
```

| **Note** |
|:---------|
| Make sure you're using `Python > 3.7`|

Create a new branch for your project:

```bash
git checkout -b my_project  # replace this with your project name
```

| **Note** |
|:---------|
| For [MLOps Community Engineering Labs Hackathon](https://flyte.org/hackathon/) participants: Each team will have its own branch on the main `flyteorg/flytelab` repo. If you're part of a team of more than one person, assign *one teammate* to create a project directory and push it into your team's branch. |

We use `cookiecutter` to manage project templates.

Install prerequisites:

```
pip install cookiecutter
```

In the root of the repo, create a new project:

```bash
cookiecutter templates/basic -o projects
```

| **Note** |
|:---------|
| There are more templates in the `templates` directory depending on the requirements of your project. |

Answer the project setup questions:

```
project_name: my_project          # replace this with your project name (can only contain alphanumeric characters, `.`, and `_`)
project_author: foobar            # replace this with your name
github_username: my_username      # replace this with your github username
description: project description  # optional
```

| **Note** |
|:---------|
| For [MLOps Community Engineering Labs Hackathon](https://flyte.org/hackathon/) participants: `project_author` should be your team name. |

The project structure looks like the following:
```bash
.
├── Dockerfile
├── README.md
├── dashboard
│   ├── app.py  # streamlit app
│   ├── remote.config
│   └── sandbox.config
├── deploy.py  # deployment script
├── my_project
│   ├── __init__.py
│   └── workflows.py  # flyte workflows
├── requirements-dev.txt
└── requirements.txt
```

## 🌏 Environment Setup

Go into the project directory, then create your project's virtual environment:

```bash
cd projects/my_project

# create and activate virtual environment
python -m venv env
source env/bin/activate

# install requirements
pip install -r requirements.txt -r requirements-dev.txt
```

Run Flyte workflows locally:

```
python my_project/workflows.py
```

You should see something like this in the output (you can ignore the warnings):
```
trained model: LogisticRegression()
```

Congrats! You just setup your flytelab project 🌟.

You can now modify and iterate on the `workflows.py` file to create your very own Flyte
workflows using `flytekit`. You can refer to the
[User Guide](https://docs.flyte.org/projects/cookbook/en/latest/index.html),
[Tutorials](https://docs.flyte.org/projects/cookbook/en/latest/tutorials.html),
and [Flytekit API Reference](https://docs.flyte.org/projects/flytekit/en/latest/) to
learn more about all of `Flyte`'s capabilities.

## 🚢 Deployment

So far you've probably been running your workflows locally by invoking `python my_project/workflows.py`.
The first step to deploying your workflows to a Flyte cluster is to test it out on a
[local sandbox cluster](https://docs.flyte.org/en/latest/deployment/sandbox.html).

Make sure you have [docker](https://docs.docker.com/get-docker/) installed.

Then install `flytectl`:

<details>

<summary>OSX</summary>

```bash
brew install flyteorg/homebrew-tap/flytectl
```

</details>

<details>

<summary>Other Operating Systems</summary>

```bash
curl -sL https://ctl.flyte.org/install | sudo bash -s -- -b /usr/local/bin # You can change path from /usr/local/bin to any file system path
export PATH=$(pwd)/bin:$PATH # Only required if user used different path then /usr/local/bin
```

</details>

### Sandbox Deployment

Start the sandbox cluster:

```bash
flytectl sandbox start --source .
```

| **Note** |
|:---------|
| If you're having trouble getting the Flyte sandbox to start, see the [troubleshooting guide](https://docs.flyte.org/en/latest/community/troubleshoot.html#troubleshooting-guide). |

You should now be able to go to `http://localhost:30081/console` on your browser to see the Flyte UI.

`git commit` your changes, then deploy your project's workflows with:

```bash
python deploy.py
```

<details>

<summary>Expected output</summary>

You should see something like:

```
Successfully packaged 4 flyte objects into /Users/nielsbantilan/git/flytelab/projects/my_project/flyte-package.tgz
Registering Flyte workflows
 ---------------------------------------------------------------- --------- ------------------------------
| NAME (4)                                                       | STATUS  | ADDITIONAL INFO              |
 ---------------------------------------------------------------- --------- ------------------------------
| /tmp/register724861421/0_my_project.workflows.get_dataset_1.pb | Success | Successfully registered file |
 ---------------------------------------------------------------- --------- ------------------------------
| /tmp/register724861421/1_my_project.workflows.train_model_1.pb | Success | Successfully registered file |
 ---------------------------------------------------------------- --------- ------------------------------
| /tmp/register724861421/2_my_project.workflows.main_2.pb        | Success | Successfully registered file |
 ---------------------------------------------------------------- --------- ------------------------------
| /tmp/register724861421/3_my_project.workflows.main_3.pb        | Success | Successfully registered file |
 ---------------------------------------------------------------- --------- ------------------------------
4 rows
```

</details>

On the Flyte UI, you'll see a `flytelab-<project-name>` project namespace on the homepage.
Navigate to the `my_project.workflows.main` workflow and hit the `Launch Workflow` button, then
the `Launch` button on the model form.

🎉 Congrats! You just kicked off your first workflow on your local Flyte sandbox cluster.

<!-- TODO: add instructions for fast registration -->

### Union.ai Playground Deployment

The [Union.ai](https://union.ai/) team maintains a playground Flyte cluster that you can use
to run your workflows.

When you're ready to deploy your workflows to a full-fledged production Flyte cluster, first you'll need to
request an account on the Flyte OSS Slack [`#flytelab` channel](https://flyte-org.slack.com/archives/C032ZU3FSAX).

| **Note** |
|:---------|
| For [MLOps Community Engineering Labs Hackathon](https://flyte.org/hackathon/) participants: you will receive these credentials after all teams have been finalized. |

You'll receive a `username` and `password` to sign into the [Union.ai Playground](https://playground.hosted.unionai.cloud/console), in addition to a `client_id` and `client_secret` if you want to use the [FlyteRemote](https://docs.flyte.org/projects/flytekit/en/latest/design/control_plane.html#design-control-plane) object to get the input and output data of your workflow executions from the playground.

#### Hosting Docker Images on Github Container Registry

Create a [personal access token (PAT)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) on github.
Make sure to give your PAT [read and write access to packages](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry)

Then authenticate to the `ghcr.io` registry:

```bash
export CR_PAT="<your-token>"
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
```

Go to `https://github.com/<your-username>/flytelab/pkgs/container/flytelab`
and you should see a package called `flytelab`, then:

1. Click **Add Repository** to link your fork of the `flytelab` repo.
2. Scroll down to the **Danger Zone**, click **Change visibility**, and make the package public.

Then, deploying to the playground is as simple as:

```
python deploy.py --remote
```

Go to https://playground.hosted.unionai.cloud, authenticate with your union.ai playground
`username` and `password`, where you can navigate to your `flytelab-<project-name>` project
to run your workflows.

<!-- TODO: add instructions for fast registration -->


## 💻 Streamlit App [Optional]

The `basic` project template ships with a `dashboard/app.py` script that uses
[`streamlit`](https://streamlit.io/) as a UI for interacting with your model.

```
pip install streamlit
```

### Run App Locally against Sandbox Cluster

```
streamlit run dashboard/app.py
```

### Run App Locally against Union.ai Playground Cluster

To access the data on the Union.ai playground, first export your `client_id` and `client_secret`
to your terminal session.

```
export FLYTE_CREDENTIALS_CLIENT_ID="<client_id>"
export FLYTE_CREDENTIALS_CLIENT_SECRET="<client_secret>"
```

Then start serving your streamlit app with:

```
streamlit run dashboard/app.py -- --remote
```

### Deploying to Streamlit Cloud

If you want to use [streamlit cloud](https://streamlit.io/cloud) to deploy your app
to share with the world, push your changes to the remote github branch you're working
from and point streamlit cloud to the streamlit app script:

```
flytelab/projects/my_project/dashboard/app.py
```

You'll need to use their [Secrets management](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management) system on the streamlit cloud UI
to add your client id and secret credentials so that it has access to the playground
cluster:

```bash
FLYTE_BACKEND = "remote"  # point the app to the playground backend
FLYTE_CREDENTIALS_CLIENT_ID = "<client_id>"  # replace this with your client id
FLYTE_CREDENTIALS_CLIENT_SECRET = "<client_secret>"  # replace this with your client secret
```

You can also add additional secrets to the secrets file if needed.
