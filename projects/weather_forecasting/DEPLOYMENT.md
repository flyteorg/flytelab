# Deployment Instructions

This page contains notes regarding how to use Flyte in an end-to-end ML
system.

## Project Setup

### Sandbox

<!-- TODO update project registration and resource updating to use flytectl -->

```bash
flyte-cli register-project -h localhost:30081 -i -p flytelab -n flytelab -d 'ML projects using Flyte'
```

```bash
flyte-cli -i \
    -h localhost:30081 \
    -p flytelab \
    -d development update-cluster-resource-attributes \
    --attributes projectQuotaCpu 16 \
    --attributes projectQuotaMemory 30Gi
```

### Remote

<!-- TODO update project registration and resource updating to use flytectl -->

```bash
flyte-cli register-project -h demo.nuclyde.io -p flytelab -n flytelab -d 'ML projects using Flyte'
```

```bash
flyte-cli \
    -h demo.nuclyde.io \
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

## Production [demo.nuclyde.io](https://demo.nuclyde.io/console)

Make sure you have a config file `~/.flyte/nuclydedemo-config.yaml` in your local filesystem:

```
admin:
  # For GRPC endpoints you might want to use dns:///flyte.myexample.com
  endpoint: dns:///demo.nuclyde.io
  authType: Pkce
  # Change insecure flag to ensure that you use the right setting for your environment
  insecure: false
storage:
  type: stow
  stow:
    kind: s3
    config:
      auth_type: iam
      region: us-east-2
  container: flyte-demo
logger:
  # Logger settings to control logger output. Useful to debug logger:
  show-source: true
  level: 1

```

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
