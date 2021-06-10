# Development Notes

This page contains notes regarding how to use Flyte in an end-to-end ML
system.

## Project Setup

### Sandbox

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

```bash
flyte-cli register-project -h sandbox.uniondemo.run -p flytelab -n flytelab -d 'ML projects using Flyte'
```

```bash
flyte-cli \
    -h sandbox.uniondemo.run \
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
INSECURE=true \
FLYTE_HOST=localhost:30081 \
FLYTE_CONFIG=.flyte/sandbox.config \
OUTPUT_DATA_PREFIX=s3://my-s3-bucket \
REGISTRY=public.ecr.aws/r1m6i5g8 \
make register
```

### Fast Registering New Code

In case you've only changed user code and not system-level dependencies:

```bash
FLYTE_AWS_ENDPOINT=http://localhost:30084/ \
FLYTE_AWS_ACCESS_KEY_ID=minio \
FLYTE_AWS_SECRET_ACCESS_KEY=miniostorage \
INSECURE=true \
FLYTE_HOST=localhost:30081 \
FLYTE_CONFIG=.flyte/sandbox.config \
OUTPUT_DATA_PREFIX=s3://my-s3-bucket \
ADDL_DISTRIBUTION_DIR=s3://my-s3-bucket/cookbook \
REGISTRY=public.ecr.aws/r1m6i5g8 \
make fast_register
```

## Production [sandbox.uniondemo.run](https://sandbox.uniondemo.run/console)

### Register Workflows

```bash
FLYTE_HOST=sandbox.uniondemo.run \
FLYTE_CONFIG=.flyte/remote.config \
SERVICE_ACCOUNT=default \
OUTPUT_DATA_PREFIX=s3://flytelab/raw_data \
REGISTRY=public.ecr.aws/r1m6i5g8 \
make register
```

### Fast Registering New Code

```bash
FLYTE_HOST=sandbox.uniondemo.run \
FLYTE_CONFIG=.flyte/remote.config \
SERVICE_ACCOUNT=default \
OUTPUT_DATA_PREFIX=s3://flytelab/raw_data \
ADDL_DISTRIBUTION_DIR=s3://flytelab/fast \
REGISTRY=public.ecr.aws/r1m6i5g8 \
make fast_register
```

### Activating Launch Plans

List launch plan versions

```bash
flyte-cli -h sandbox.uniondemo.run -p flytelab -d development list-launch-plan-versions
```

Get the `urn` of the launch plan you want to activate, e.g. `lp:flytelab:development:atlanta_weather_forecast:2aa94baac33217d4c89685946a8e434b15d48f3a`

```bash
flyte-cli update-launch-plan -h sandbox.uniondemo.run --state active -u lp:flytelab:development:atlanta_weather_forecast:2aa94baac33217d4c89685946a8e434b15d48f3a
```


### Test NOAA API

This makes sure that a docker image is able to call the NOAA API and return a valid result.

```
docker run --rm public.ecr.aws/r1m6i5g8/flytelab:weather-forecasting-latest python -c 'import requests; url = "https://www.ncei.noaa.gov/access/services/search/v1/data?dataset=global-hourly&bbox=33.886823,-84.551068,33.647808,-84.28956&startDate=2021-05-10&endDate=2021-05-11&units=metric&format=json&limit=1000&offset=0"; r = requests.get(url); print(r.text)'
```
