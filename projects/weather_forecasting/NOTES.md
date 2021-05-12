# Development Notes

This page contains notes regarding how to use Flyte in an end-to-end ML
system.

## Sandbox

### Register Workflows

```
INSECURE=true FLYTE_HOST=localhost:30081 OUTPUT_DATA_PREFIX=s3://my-s3-bucket ADDL_DISTRIBUTION_DIR=s3://my-s3-bucket/flyte-fast-distributions REGISTRY=public.ecr.aws/nuclyde make register
```

### Fast Registering New Code

In case you've only changed user code and not system-level dependencies:

```
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

```
FLYTE_HOST=demo.nuclyde.io \
OUTPUT_DATA_PREFIX=s3://flyte-demo/raw_data \
ADDL_DISTRIBUTION_DIR=s3://flyte-demo/tars \
REGISTRY=public.ecr.aws/nuclyde \
make register
```

### Fast Registering New Code

```
FLYTE_HOST=demo.nuclyde.io \
OUTPUT_DATA_PREFIX=s3://flyte-demo/raw_data \
ADDL_DISTRIBUTION_DIR=s3://flyte-demo/tars \
REGISTRY=public.ecr.aws/nuclyde \
make fast_register
```
