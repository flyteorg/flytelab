export REPOSITORY=flytelab

VERSION=$(shell git rev-parse HEAD)
IMAGE_NAME=flytelab

ifeq ($(INSECURE), true)
	INSECURE=-i
endif

ifeq ($(NOPUSH), true)
	NOPUSH=1
endif

ifndef FLYTECTL_CONFIG
	FLYTECTL_CONFIG=~/.flyte/config.yaml
endif


# If the REGISTRY environment variable has been set, that means the image name will not just be tagged as
#   flytecookbook:<sha> but rather,
#   docker.io/lyft/flytecookbook:<sha> or whatever your REGISTRY is.
ifneq ($(origin REGISTRY), undefined)
	FULL_IMAGE_NAME = ${REGISTRY}/${IMAGE_NAME}
else
	FULL_IMAGE_NAME = ${IMAGE_NAME}
endif

export FLYTE_HOST ?= localhost:30081
export PREFIX ?= weather-forecasting
export FLYTE_CONFIG ?= .flyte/sandbox.config

# The Flyte project and domain that we want to register under
export PROJECT ?= flytelab
export DOMAIN ?= development
export DESCRIPTION ?= 'ML projects using Flyte'

# This specifies where fast-registered code is uploaded to during registration.
# If you're not using the standard minio deployment on flyte sandbox: update this path to something that
#   - you have write access to
#   - flytepropeller can read (depending on the role it uses)
export ADDL_DISTRIBUTION_DIR ?= s3://my-s3-bucket/flyte-fast-distributions

FLYTE_INTERNAL_IMAGE=${FULL_IMAGE_NAME}:${PREFIX}-${VERSION}
FLYTE_INTERNAL_LATEST=${FULL_IMAGE_NAME}:${PREFIX}-latest

# targets for local development
venv:
	@virtualenv ./.venv/weather-forecasting

deps:
	@pip install -r requirements.txt

env.txt:
	@echo "NOAA_API_KEY='<API_KEY>'" > env.txt

.PHONY: env-export
env-export:
	@eval $(sed 's/^/export /g' env.txt)

# flyte-related targets
.PHONY: create-project
create-project:
	flyte-cli register-project -h ${FLYTE_HOST} ${INSECURE} -p ${PROJECT} -n ${PROJECT} -d ${DESCRIPTION}

.PHONY: _requires-commit
_requires-commit:
	@if [ -n "$(shell git status --porcelain)" ]; then \
		echo "Please commit git changes before building"; \
		exit 1; \
	fi;

.PHONY: docker-build
docker-build: _requires-commit
ifndef NOAA_API_KEY
	$(error NOAA_API_KEY must be defined)
endif
	@echo "Building: ${FLYTE_INTERNAL_IMAGE}"
	docker build . \
		--build-arg tag="${FLYTE_INTERNAL_IMAGE}" \
		--build-arg config="${FLYTE_CONFIG}" \
		--build-arg noaa_api_key="${NOAA_API_KEY}" \
		-t "${FLYTE_INTERNAL_IMAGE}" \
		-t "${FLYTE_INTERNAL_LATEST}" \
		-f ./Dockerfile

.PHONY: docker-push
docker-push: docker-build
	@echo "Pushing: ${FLYTE_INTERNAL_IMAGE}"
	docker push "${FLYTE_INTERNAL_IMAGE}"
	docker push "${FLYTE_INTERNAL_LATEST}"

.PHONY: serialize
serialize:
	echo ${CURDIR}
	rm -rf ${CURDIR}/_pb_output || true
	mkdir ${CURDIR}/_pb_output || true
	pyflyte -c flyte.config --pkgs app package \
		--force \
		--in-container-source-path /root \
		--image ${FULL_IMAGE_NAME}:${PREFIX}-${VERSION}

.PHONY: register
register: docker-push serialize
	flytectl -c ${FLYTECTL_CONFIG} \
		register files \
		--project flytelab \
		--domain development \
		--archive flyte-package.tgz \
		--force \
		--version ${VERSION}

.PHONY: fast_serialize
fast_serialize:
	echo ${CURDIR}
	mkdir ${CURDIR}/_pb_output || true
	rm ${CURDIR}/_pb_output/* || true
	pyflyte -c flyte.config --pkgs app package \
		--force \
		--in-container-source-path /root \
		--fast \
		--image ${FLYTE_INTERNAL_LATEST}

.PHONY: fast_register
fast_register: fast_serialize
	flytectl -c ${FLYTECTL_CONFIG}} \
		register files \
		--project flytelab \
		--domain development \
		--archive flyte-package.tgz \
		--version fast${VERSION}
