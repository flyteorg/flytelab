FROM python:3.9-slim-buster

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

# e.g. flyte.config or sandbox.config
ARG config


RUN apt-get update && \
    apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    build-essential

# Install the AWS cli separately to prevent issues with boto being written over
RUN pip3 install awscli

ENV VENV /opt/venv

# Virtual environment
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /root
RUN pip install -r /root/requirements.txt

COPY whats_cooking_good_looking /root/whats_cooking_good_looking
COPY $config /root/flyte.config


ARG creds
RUN echo $creds | base64 -d > /root/google_creds.json
ENV GOOGLE_APPLICATION_CREDENTIALS "/root/google_creds.json"

# This image is supplied by the build script and will be used to determine the version
# when registering tasks, workflows, and launch plans
ARG image
ENV FLYTE_INTERNAL_IMAGE $image
