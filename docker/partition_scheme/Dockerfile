# 
# This is a Dockerfile for building OpenCV debian packages
# with CUDA, cuDNN, GStreamer, ect enabled.  You can then take
# the output .deb packages and install them into other containers.
#
# See scripts/docker_build_opencv.sh to run it
#

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.4
FROM ${BASE_IMAGE} AS opencv_base

#
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

#WORKDIR /opt


#
# OpenCV - https://github.com/mdegans/nano_build_opencv/blob/master/build_opencv.sh
#
ARG OPENCV_VERSION="4.5.5"

# install build dependencies
COPY ./opencv_install_deps.sh opencv_install_deps.sh
RUN ./opencv_install_deps.sh

# OpenCV looks for the cuDNN version in cudnn_version.h, but it's been renamed to cudnn_version_v8.h
#RUN ln -s /usr/include/$(uname -i)-linux-gnu/cudnn_version_v8.h /usr/include/$(uname -i)-linux-gnu/cudnn_version.h
#RUN ln -s /usr/include/$(uname -i)-linux-gnu/cudnn.h /usr/include/cudnn.h
RUN echo $(ls -lh /usr/include/$(uname -i)-linux-gnu)

# architecture-specific build options
ARG CUDA_ARCH_BIN=""
ARG ENABLE_NEON="OFF"

FROM opencv_base AS build1
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone && \
    apt-get update && \
    apt-get install -y vim cmake g++ unzip git

# install opencv on Jetson nano
#FROM BASIC_BUILDX AS BUILD1
#RUN apt-get update && \
#    apt-get install -y build-essential apt-utils && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -yq libopencv-dev && \
#    apt-get install -y libatlas-base-dev

#FROM BASIC_BUILD AS BUILD1
#COPY buildOpenCV.sh /
#RUN  /buildOpenCV.sh

#FROM BASIC_BUILDD AS BUILD1
ENV CUDA_HOME="/usr/local/cuda/"
ENV PATH="/usr/local/cuda/bin:${PATH}"
#COPY opencv.zip /
#COPY opencv_contrib.zip /
RUN  sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
RUN  ldconfig
RUN  apt-get install -y build-essential cmake git unzip pkg-config zlib1g-dev && \
     apt-get install -y libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev && \
     apt-get install -y libpng-dev libtiff-dev libglew-dev && \
     apt-get install -y libavcodec-dev libavformat-dev libswscale-dev && \
     apt-get install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk* && \
     apt-get install -y python-dev python-numpy python-pip && \
     apt-get install -y python3-dev python3-numpy python3-pip && \
     apt-get install -y libxvidcore-dev libx264-dev libgtk-3-dev && \
     apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev && \
     apt-get install -y gstreamer1.0-tools libgstreamer-plugins-base1.0-dev && \
     apt-get install -y libgstreamer-plugins-good1.0-dev && \
     apt-get install -y libv4l-dev v4l-utils v4l2ucp qv4l2 && \
     apt-get install -y libtesseract-dev libxine2-dev libpostproc-dev && \
     apt-get install -y libavresample-dev libvorbis-dev && \
     apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev && \
     apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev && \
     apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev && \
     apt-get install -y liblapack-dev liblapacke-dev libeigen3-dev gfortran && \
     apt-get install -y libhdf5-dev libprotobuf-dev protobuf-compiler && \
     apt-get install -y libgoogle-glog-dev libgflags-dev
#RUN /bin/bash -c "ls -lh /"
#RUN cd / && \
#    wget -O ~/opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip && \
#    wget -O ~/opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
#RUN unzip /opencv.zip -d / && unzip /opencv_contrib.zip -d / && \
#    mv /opencv-4.5.5 /opencv && mv /opencv_contrib-4.5.5 /opencv_contrib
#RUN /bin/bash -c "ls -lh ~/"
#RUN cd /opencv && \ 
#    mkdir build && cd build && \
#    cmake -D CMAKE_BUILD_TYPE=RELEASE \
#    -D CMAKE_INSTALL_PREFIX=/usr/local \
#    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
#    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
#    -D WITH_OPENCL=OFF \
#    -D WITH_CUDA=ON \
#    -D CUDA_ARCH_BIN=5.3 \
#    -D CUDA_ARCH_PTX="" \
#    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 \
#    -D WITH_CUDNN=ON \
#    -D WITH_CUBLAS=ON \
#    -D ENABLE_FAST_MATH=ON \
#    -D CUDA_FAST_MATH=ON \
#    -D OPENCV_DNN_CUDA=ON \
#    -D ENABLE_NEON=ON \
#    -D WITH_QT=OFF \
#    -D WITH_OPENMP=ON \
#    -D BUILD_TIFF=ON \
#    -D WITH_FFMPEG=ON \
#    -D WITH_GSTREAMER=ON \
#    -D WITH_TBB=ON \
#    -D BUILD_TBB=ON \
#    -D BUILD_TESTS=OFF \
#    -D WITH_EIGEN=ON \
#    -D WITH_V4L=ON \
#    -D WITH_LIBV4L=ON \
#    -D OPENCV_ENABLE_NONFREE=ON \
#    -D INSTALL_C_EXAMPLES=OFF \
#    -D INSTALL_PYTHON_EXAMPLES=OFF \
#    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
#    -D OPENCV_GENERATE_PKGCONFIG=ON \
#    -D BUILD_EXAMPLES=OFF ..
#RUN make -j4 && \
#    rm -r /usr/include/opencv4/opencv2 && \
#    make install && \
#    ldconfig && \
#    make clean && \
#    apt-get update


# install libedgetpu, reference: google-coral/libedgetpu/docker/Dockerfile
FROM build1 AS build2
COPY update_sources.sh /
RUN /update_sources.sh
RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
 
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#    libc6-dev:arm64 
#    libc6-dev:armhf 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-all \
    python3-numpy \
    build-essential \ 
#    crossbuild-essential-armhf \
#    crossbuild-essential-arm64 \
    libusb-1.0-0-dev \
#    libusb-1.0-0-dev:arm64 \
#    libusb-1.0-0-dev:armhf \
    zlib1g-dev \
#    zlib1g-dev:armhf \
#    zlib1g-dev:arm64 \
    sudo \
    debhelper \
    pkg-config \
    zip \
    unzip \
    curl \
    wget \
    git \
    tree \
    software-properties-common \
    $(grep Ubuntu /etc/os-release > /dev/null && echo vim-common || echo xxd)

RUN apt-get install -y libeigen3-dev && \
    sudo ln -s /usr/include/engin3/Eigen /usr/include/Eigen
# for quick scipy.signal.fftconvolve test in python2 only
RUN apt-get install -y python-scipy 

FROM build2  AS FINAL 
# setup LD_LIBRARY_PATH
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/libedgetpu/out/direct/k8:/home/libedgetpu/out/direct/aarch64:/usr/local/cuda-10.2/lib64/" >> ~/.bashrc
WORKDIR /home


