#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

# source configure file
. "${GITTOP}/configure.cfg"
PROJ=${GPGTPU_PARTITION}

# paths setup
DOCKERFILE_PATH="${GITTOP}/docker/partition_scheme"
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME=${PROJ}_container
DATASET_DIR="${DATASET_HOST_ROOT}"
SRC_DIR="${GITTOP}"
DATASET_TARGET_DIR="${DATASET_MOUNT}" # the dataset mount point within container
SRC_TARGET_DIR="${SRC_MOUNT}" # the src code mount point within container

# build dockerfile to generate docker image
echo "[${PROJ}] - building docker image from dockerfile..."
DOCKER_BUILDKIT=1 nvidia-docker build -t ${IMAGE_NAME} ${DOCKERFILE_PATH}

nvidia-docker stop ${CONTAINER_NAME}
nvidia-docker rm ${CONTAINER_NAME}

# generate container from image
# mount dataset dir (ImageNet)/src  from host fs to container fs
# get the container running
echo "[${PROJ}] - build docker container..."
# Use 'priviledged' flag to enable edgetpu access
nvidia-docker run -d \
         -it \
         --privileged \
         -e IS_GPGTPU_CONTAINER='true' \
         --name ${CONTAINER_NAME} \
         --runtime=nvidia \
         -e NVIDIA_VISIBLE_DEVICES=all \
         --gpus all \
         --mount type=bind,source=/etc/passwd,target=/etc/passwd,readonly \
         --mount type=bind,source=/etc/group,target=/etc/group,readonly \
         -u $(id -u $USER):$(id -g $USER) \
         --mount type=bind,source=${SRC_DIR},target=${SRC_TARGET_DIR} \
         --mount type=bind,source=${GPTPU_LIB_BASE},target=${GPTPU_LIB_MOUNT} \
         --mount type=bind,source=/usr/lib/aarch64-linux-gnu,target=/usr/lib/aarch64-linux-gnu,readonly \
         ${IMAGE_NAME} \
         bash
         
