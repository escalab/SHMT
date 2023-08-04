#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

# source configure file
. "${GITTOP}/configure.cfg"
PROJ=${GPGTPU_TRAIN}

# paths setup
DOCKERFILE_PATH="${GITTOP}/docker/kernel_model_training"
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME=${PROJ}_container
DATASET_DIR="${DATASET_HOST_ROOT}"
SRC_DIR="${GITTOP}"
DATASET_TARGET_DIR="${DATASET_MOUNT}" # the dataset mount point within container
SRC_TARGET_DIR="${SRC_MOUNT}" # the src code mount point within container

# build dockerfile to generate docker image
echo "[${PROJ}] - building docker image from dockerfile..."
docker build -t ${IMAGE_NAME} ${DOCKERFILE_PATH}

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# generate container from image
# mount dataset dir (ImageNet)/src  from host fs to container fs
# get the container running
echo "[${PROJ}] - build docker container..."
docker run -d \
         -it \
         -e IS_GPGTPU_CONTAINER='true' \
         --name ${CONTAINER_NAME} \
         --gpus all \
         --mount type=bind,source=${SRC_DIR},target=${SRC_TARGET_DIR} \
         ${IMAGE_NAME} \
         bash


#--mount type=bind,source=${DATASET_DIR},target=${DATASET_TARGET_DIR} \
         






