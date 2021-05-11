export REPOSITORY=flytelab-weather-forecasting

VERSION=$(shell git rev-parse HEAD)
IMAGE_NAME=flytelab-weather-forecasting

ifeq ($(INSECURE), "true")
	INSECURE=-i
endif

ifeq ($(NOPUSH), "true")
	NOPUSH=1
endif

# If the REGISTRY environment variable has been set, that means the image name will not just be tagged as
#   flytecookbook:<sha> but rather,
#   docker.io/lyft/flytecookbook:<sha> or whatever your REGISTRY is.
ifneq ($(origin REGISTRY), undefined)
	FULL_IMAGE_NAME = ${REGISTRY}/${IMAGE_NAME}
else
	FULL_IMAGE_NAME = ${IMAGE_NAME}
endif

export FLYTE_HOST ?= demo.nuclyde.io

# The Flyte project and domain that we want to register under
export PROJECT ?= flytelab
export DOMAIN ?= development


venv:
	@virtualenv ./.venv/weather-forecasting

deps:
	@pip install -r requirements.txt

env.txt:
	@echo "NOAA_API_KEY='<API_KEY>'" > env.txt

.PHONY: env-export
env-export:
	@eval $(sed 's/^/export /g' env.txt)

.PHONY: create-project
create-project:
	flyte-cli register-project -p flytelab -n flytelab -d 'ML projects using Flyte'

.PHONY: docker-build
docker-build:
	NOPUSH=${NOPUSH} IMAGE_NAME=${IMAGE_NAME} flytekit_build_image.sh ./Dockerfile ${PREFIX}

.PHONY: serialize
serialize:
	echo ${CURDIR}
	rm -rf ${CURDIR}/_pb_output || true
	mkdir ${CURDIR}/_pb_output || true
	pyflyte -c flyte.config --pkgs app \
		serialize \
		--in-container-config-path /root/flyte.config \
		--image ${FULL_IMAGE_NAME}:${VERSION} workflows -f _pb_output

.PHONY: register
register: docker_build serialize
	flyte-cli register-files -h ${FLYTE_HOST} ${INSECURE} -p ${PROJECT} -d ${DOMAIN} -v ${VERSION} ${CURDIR}/_pb_output/*
