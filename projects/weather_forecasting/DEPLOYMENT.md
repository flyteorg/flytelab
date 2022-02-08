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


Make sure the NOAA api key is available in the shell session:
```
eval $(sed 's/^/export /g' env.txt)
```

## Workflow Registration

## Sandbox

### Register Workflows

```bash
FLYTECTL_CONFIG=.flyte/sandbox-config.yaml REGISTRY=ghcr.io/flyteorg make register
```

### Fast Registering New Code

In case you've only changed user code and not system-level dependencies:

```bash
FLYTECTL_CONFIG=.flyte/sandbox-config.yaml REGISTRY=ghcr.io/flyteorg make fast_register
```

## Production [playground.hosted.unionai.cloud](https://playground.hosted.unionai.cloud/console)

### Register Workflows

```bash
FLYTECTL_CONFIG=.flyte/remote-config.yaml FLYTE_CONFIG=.flyte/remote.config REGISTRY=ghcr.io/flyteorg make register
```

### Fast Registering New Code

```bash
FLYTECTL_CONFIG=.flyte/remote-config.yaml REGISTRY=ghcr.io/flyteorg make fast_register
```

### Activating Launch Plans

List launch plan versions

```bash
./scripts/launch-plan-status.sh
```

To activate launch plans

```bash
./scripts/activate-launch-plans.sh  # [VERSION] argument is optional to activate a specific version
```

To deactivate:

```bash
./scripts/archive-launch-plans.sh  # [VERSION] argument is optional to activate a specific version
```
