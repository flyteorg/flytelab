export REPOSITORY=flytelab

VERSION=$(shell git rev-parse HEAD)
IMAGE_NAME=flytelab

ifeq ($(INSECURE), "true")
	INSECURE=-i
endif

NOPUSH ?= 

# If the REGISTRY environment variable has been set, that means the image name will not just be tagged as
#   flytecookbook:<sha> but rather,
#   docker.io/lyft/flytecookbook:<sha> or whatever your REGISTRY is.
ifneq ($(origin REGISTRY), undefined)
	FULL_IMAGE_NAME = ${REGISTRY}/${IMAGE_NAME}
else
	FULL_IMAGE_NAME = ${IMAGE_NAME}
endif

export FLYTE_HOST ?= demo.nuclyde.io
export PREFIX ?= weather-forecasting

# The Flyte project and domain that we want to register under
export PROJECT ?= flytelab
export DOMAIN ?= development
export DESCRIPTION ?= 'ML projects using Flyte'

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
	flyte-cli register-project -p ${PROJECT} -n ${PROJECT} -d ${DESCRIPTION}

.PHONY: docker-build
docker-build:
	echo ${NOPUSH}
	# REGISTRY=${REGISTRY} NOPUSH=${NOPUSH} IMAGE_NAME=${IMAGE_NAME} flytekit_build_image.sh ./Dockerfile ${PREFIX}

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
register: docker-build serialize
	flyte-cli register-files -h ${FLYTE_HOST} ${INSECURE} -p ${PROJECT} -d ${DOMAIN} -v ${VERSION} ${CURDIR}/_pb_output/*
