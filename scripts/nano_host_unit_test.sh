#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"
BUILD_DIR="./nano_host_build"

echo ${BUILD_DIR}

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake ..
make clean
make
sudo ./gpgtpu sobel_2d 2048 100 cpu gpu
