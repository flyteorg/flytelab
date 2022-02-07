# Deployment Instructions

This page contains notes regarding how to use Flyte in an end-to-end ML
system.

## Project Setup

### Sandbox

Create project:

```bash
flytectl create project \
    --name flytelab \
    --id flytelab \
    --description "flytelab: ml projects in flyte" \
    --config .flyte/sandbox-config.yaml \
    --project flytelab
```

Update cluster resource attributes:

```bash
flyte-cli -i \
    -h localhost:30081 \
    -p flytelab \
    -d development update-cluster-resource-attributes \
    --attributes projectQuotaCpu 16 \
    --attributes projectQuotaMemory 30Gi
```

### Remote

Create project:

```bash
flytectl create project \
    --name flytelab \
    --id flytelab \
    --description "flytelab: ml projects in flyte" \
    --config .flyte/remote-config.yaml \
    --project flytelab
```

Update cluster resource attributes:

```bash
flyte-cli \
    -h playground.hosted.unionai.cloud \
    -p flytelab \
    -d development update-cluster-resource-attributes \
    --attributes projectQuotaCpu 16 \
    --attributes projectQuotaMemory 30Gi
```


Make sure the NOAA api key is available in the shell session:
```
eval $(sed 's/^/export /g' env.txt)
```

## Workflow Registration

## Sandbox

### Register Workflows

```bash
FLYTECTL_CONFIG=.flyte/sandbox-config.yaml \
SERVICE_ACCOUNT=default \
REGISTRY=ghcr.io/flyteorg \
make register
```

### Fast Registering New Code

In case you've only changed user code and not system-level dependencies:

```bash
FLYTECTL_CONFIG=.flyte/sandbox-config.yaml \
SERVICE_ACCOUNT=default \
REGISTRY=ghcr.io/flyteorg \
make fast_register
```

## Production [playground.hosted.unionai.cloud](https://playground.hosted.unionai.cloud/console)

### Register Workflows

```bash
FLYTECTL_CONFIG=.flyte/remote-config.yaml \
SERVICE_ACCOUNT=demo \
REGISTRY=ghcr.io/flyteorg \
make register
```

### Fast Registering New Code

```bash
FLYTECTL_CONFIG=.flyte/remote-config.yaml \
SERVICE_ACCOUNT=demo \
REGISTRY=ghcr.io/flyteorg \
make fast_register
```

### Activating Launch Plans

List launch plan versions

```bash
flytectl -c .flyte/remote-config.yaml \
    get launchplan \
    -p flytelab \
    -d development \
    -o yaml \
    --latest \
    atlanta_weather_forecast_v2
```

Get the `version` of the launch plan you want to activate, then:

```bash
flytectl -c .flyte/remote-config.yaml \
    update launchplan \
    -p flytelab \
    -d development \
    atlanta_weather_forecast_v2 \
    --version <version> \
    --activate
```

To deactivate:

```bash
flytectl -c .flyte/remote-config.yaml \
    update launchplan \
    -p flytelab \
    -d development \
    atlanta_weather_forecast_v2 \
    --version <version> \
    --archive
```

### Test NOAA API

This makes sure that a docker image is able to call the NOAA API and return a valid result.

```
docker run --rm public.ecr.aws/r1m6i5g8/flytelab:weather-forecasting-latest python -c 'import requests; url = "https://www.ncei.noaa.gov/access/services/search/v1/data?dataset=global-hourly&bbox=33.886823,-84.551068,33.647808,-84.28956&startDate=2021-05-10&endDate=2021-05-11&units=metric&format=json&limit=1000&offset=0"; r = requests.get(url); print(r.text)'
```
