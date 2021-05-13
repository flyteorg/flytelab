# Development Notes

This page contains notes regarding how to use Flyte in an end-to-end ML
system.

## Sandbox

### Register Workflows

```bash
INSECURE=true \
FLYTE_HOST=localhost:30081 \
FLYTE_CONFIG=.flyte/sandbox.config \
SERVICE_ACCOUNT=default \
OUTPUT_DATA_PREFIX=s3://my-s3-bucket \
ADDL_DISTRIBUTION_DIR=s3://my-s3-bucket/flyte-fast-distributions \
REGISTRY=public.ecr.aws/nuclyde make register
```

### Fast Registering New Code

In case you've only changed user code and not system-level dependencies:

```bash
INSECURE=true \
FLYTE_HOST=localhost:30081 \
FLYTE_CONFIG=.flyte/sandbox.config \
SERVICE_ACCOUNT=default \
OUTPUT_DATA_PREFIX=s3://my-s3-bucket \
ADDL_DISTRIBUTION_DIR=s3://my-s3-bucket/flyte-fast-distributions \
REGISTRY=public.ecr.aws/nuclyde \
make fast_register
```

## Production [demo.nuclyde.io](https://demo.nuclyde.io/console)

### Register Workflows

```bash
FLYTE_HOST=demo.nuclyde.io \
FLYTE_CONFIG=.flyte/remote.config \
SERVICE_ACCOUNT=demo \
OUTPUT_DATA_PREFIX=s3://flyte-demo/raw_data \
ADDL_DISTRIBUTION_DIR=s3://flyte-demo/tars \
REGISTRY=public.ecr.aws/nuclyde \
make register
```

### Fast Registering New Code

```bash
FLYTE_HOST=demo.nuclyde.io \
FLYTE_CONFIG=.flyte/remote.config \
SERVICE_ACCOUNT=demo \
OUTPUT_DATA_PREFIX=s3://flyte-demo/raw_data \
ADDL_DISTRIBUTION_DIR=s3://flyte-demo/tars \
REGISTRY=public.ecr.aws/nuclyde \
make fast_register
```


### Updating Cluster Resource Attributes

```bash
flyte-cli -h demo.nuclyde.io -p flytelab -d development update-cluster-resource-attributes --attributes projectQuotaCpu 1000 --attributes projectQuotaMemory 5000Gi
```
