#!/bin/sh
cd ../edgetpu
make DOCKER_IMAGE=ubuntu:18.04 DOCKER_CPUS="aarch64" DOCKER_TARGETS=examples docker-build
