#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_docker_build"

echo ${BUILD_DIR}

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake ..
make clean
make
./gpgtpu sobel_2d 1024 1 cpu tpu
