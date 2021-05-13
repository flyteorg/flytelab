# Development Notes

This page contains notes regarding how to use Flyte in an end-to-end ML
system.

## Sandbox

### Register Workflows

```bash
INSECURE=true \
FLYTE_HOST=localhost:30081 \
OUTPUT_DATA_PREFIX=s3://my-s3-bucket \
ADDL_DISTRIBUTION_DIR=s3://my-s3-bucket/flyte-fast-distributions \
REGISTRY=public.ecr.aws/nuclyde make register
```

### Fast Registering New Code

In case you've only changed user code and not system-level dependencies:

```bash
FLYTE_AWS_ENDPOINT=http://localhost:30084/ \
FLYTE_AWS_ACCESS_KEY_ID=minio \
FLYTE_AWS_SECRET_ACCESS_KEY=miniostorage \
INSECURE=true \
FLYTE_HOST=localhost:30081 \
OUTPUT_DATA_PREFIX=s3://my-s3-bucket \
ADDL_DISTRIBUTION_DIR=s3://my-s3-bucket/flyte-fast-distributions \
REGISTRY=public.ecr.aws/nuclyde \
make fast_register
```

## Production [demo.nuclyde.io](https://demo.nuclyde.io/console)

### Register Workflows

```bash
FLYTE_HOST=demo.nuclyde.io \
OUTPUT_DATA_PREFIX=s3://flyte-demo/raw_data \
ADDL_DISTRIBUTION_DIR=s3://flyte-demo/tars \
REGISTRY=public.ecr.aws/nuclyde \
make register
```

### Fast Registering New Code

```bash
FLYTE_HOST=demo.nuclyde.io \
OUTPUT_DATA_PREFIX=s3://flyte-demo/raw_data \
ADDL_DISTRIBUTION_DIR=s3://flyte-demo/tars \
REGISTRY=public.ecr.aws/nuclyde \
make fast_register
```


### Updating Cluster Resource Attributes

```bash
flyte-cli -h demo.nuclyde.io -p flytelab -d development update-cluster-resource-attributes --attributes projectQuotaCpu 1000 --attributes projectQuotaMemory 5000Gi
```
