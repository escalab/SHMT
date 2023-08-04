# Official tensorflow images built after Sept 2021 are based on Ubuntu 20.04
# Check: https://hub.docker.com/r/tensorflow/tensorflow/

FROM tensorflow/tensorflow:2.9.2-gpu AS BASE_BUILD

# edgetpu-compiler setup
FROM BASE_BUILD AS BUILD1
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    gnupg \
    ca-certificates \
    curl \
    apt-utils \
    apt-transport-https && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && apt-get install -y edgetpu-compiler

# install opencv
FROM BUILD1 AS BUILD2
RUN apt update && apt install -y libopencv-dev python3-opencv

# GPGTPU project setup (including kernel model training setup)
FROM BUILD2 AS BUILD3
RUN apt-get update && apt-get -y install vim && \
        apt-get install -y git && \
        python -m pip install scipy && \
        pip install Pillow scikit-image && \
        apt-get install -y cmake && \
        pip install -q tensorflow-model-optimization

# convention setup
FROM BUILD3 AS BUILD4
WORKDIR /home
RUN ln -s $(which python3) /usr/bin/python
