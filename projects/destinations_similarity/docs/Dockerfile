FROM python:3.7.12-slim-buster

RUN apt-get update && \
  pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir sphinx==4.2.0 sphinx_rtd_theme==1.0.0

WORKDIR /home

CMD ["sh","-c","rm -rf _build/ && sphinx-build -b html . _build/"]