# syntax=docker/dockerfile:1

# Dockerfile that sets up the Flyte image

ARG image
ARG config

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Defining environment variables
ENV APPUSER="flyte"
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"
ENV PATH="/home/${APPUSER}/.local/bin:${PATH}"
ENV PYTHONPATH="/home/${APPUSER}:${PYTHONPATH}"

# Updating and cleaning system
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential git && \
    apt-get autoremove -yqq --purge && \
    apt-get clean

# Changing the user so that the container is non-root
RUN useradd -u 1024 -m "${APPUSER}"
USER "${APPUSER}"
WORKDIR "/home/${APPUSER}"

# Setup virtual environment
ENV VENV="/home/${APPUSER}/venv"
RUN python -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

# Copy requirements
COPY requirements.txt "/home/${APPUSER}/requirements.txt"

# Upgrade pip, install dependencies and awscli
RUN python -m pip install -U pip && \
    pip install -r requirements.txt awscli

# Copy the code and configuration
COPY --chown="${APPUSER}:${APPUSER}" \
    destinations_similarity/ "/home/${APPUSER}/destinations_similarity"
COPY $config "/home/${APPUSER}/flyte.config"

# Download nltk files
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Tag the image
ENV FLYTE_INTERNAL_IMAGE="$image"
